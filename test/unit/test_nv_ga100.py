import ctypes
import struct
from types import SimpleNamespace

import pytest

from tinygrad.runtime import ops_nv
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.nv.ip import GRBufDesc, NV_FLCN_GA100, NV_GSP, ga100_gsp_userd_layout, gsp_fw_heap_size, parse_riscv_ucode_desc
from tinygrad.runtime.support.nv.nvdev import decode_gp102_lmr_vram_mib, get_nv_chip_config, require_ga100_vram_size


def test_ga100_chip_config_uses_tu102_gsp():
  config = get_nv_chip_config(0x17, 0x00)

  assert config.name == "GA100"
  assert config.boot_fw_name == "ga100"
  assert config.gsp_fw_name == "tu102"
  assert config.gsp_signature_section == ".fwsignature_ga100"
  assert config.uses_fwsec_frts is False
  assert config.frts_size == 0
  assert config.fixed_fw_heap_size == 0


def test_ga102_chip_config_is_unchanged():
  config = get_nv_chip_config(0x17, 0x02)

  assert config.name == "GA102"
  assert config.boot_fw_name == "ga102"
  assert config.gsp_fw_name == "ga102"
  assert config.gsp_signature_section == ".fwsignature_ga10x"
  assert config.uses_fwsec_frts is True
  assert config.frts_size == 1 << 20
  assert config.fixed_fw_heap_size == 0x8100000


def test_ga100_vram_size_decodes_inherited_gp102_local_memory_range():
  assert decode_gp102_lmr_vram_mib(0x208) == 8192
  assert decode_gp102_lmr_vram_mib(0x20B) == 65536
  assert decode_gp102_lmr_vram_mib(0x28B) == 81920
  assert decode_gp102_lmr_vram_mib(0x40000208) == 7680


def test_ga100_vram_preflight_is_fail_closed():
  ga100, ga102 = get_nv_chip_config(0x17, 0x00), get_nv_chip_config(0x17, 0x02)

  require_ga100_vram_size(ga100, 8192, 8192)
  require_ga100_vram_size(ga102, 0, 0)
  with pytest.raises(RuntimeError, match="requires NV_EXPECTED_VRAM_MIB"):
    require_ga100_vram_size(ga100, 8192, 0)
  with pytest.raises(RuntimeError, match="expected 65536 MiB, capacity source reports 8192 MiB"):
    require_ga100_vram_size(ga100, 8192, 65536)


def test_ga100_version4_riscv_descriptor():
  words = [4, 0, 0x488, 0x488, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  desc = parse_riscv_ucode_desc(struct.pack("<15I", *words), 0, 0x1000)

  assert desc.version == 4
  assert (desc.bootloader_offset, desc.bootloader_size) == (0, 0x488)
  assert (desc.bootloader_param_offset, desc.bootloader_param_size) == (0x488, 0x10)
  assert desc.app_version == 0
  assert desc.manifest_offset == 0
  assert desc.monitor_data_offset == 0
  assert desc.monitor_code_offset == 0


def test_ga102_version5_riscv_descriptor_uses_same_prefix_layout():
  words = [5, 0x5000, 0x880, 0x5880, 0x10, 0, 0, 0, 0, 0x800, 0x800, 0x1000, 0x1800, 0x2900, 1]
  desc = parse_riscv_ucode_desc(struct.pack("<15I", *words), 0, 0x6000)

  assert desc.version == 5
  assert (desc.bootloader_offset, desc.bootloader_size) == (0x5000, 0x880)
  assert (desc.manifest_offset, desc.manifest_size) == (0, 0x800)
  assert (desc.monitor_data_offset, desc.monitor_data_size) == (0x800, 0x1000)
  assert (desc.monitor_code_offset, desc.monitor_code_size) == (0x1800, 0x2900)


def test_riscv_descriptor_rejects_payload_overflow():
  words = [4, 0xF00, 0x200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  with pytest.raises(RuntimeError, match="bootloader range exceeds"):
    parse_riscv_ucode_desc(struct.pack("<15I", *words), 0, 0x1000)


def test_ga100_uses_ampere_a_classes():
  ga100 = get_nv_chip_config(0x17, 0x00)
  ga102 = get_nv_chip_config(0x17, 0x02)

  assert (ga100.gpfifo_class, ga100.compute_class, ga100.dma_class) == (
    nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_A, nv_gpu.AMPERE_DMA_COPY_A)
  assert (ga102.gpfifo_class, ga102.compute_class, ga102.dma_class) == (
    nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B)


def test_ga100_nvd_uses_sm80_renderer_target():
  assert ops_nv.nv_renderer_arch(0x802) == "sm_80"
  assert ops_nv.nv_renderer_arch(0x806) == "sm_86"
  assert ops_nv.nv_renderer_arch(0x809) == "sm_89"
  assert ops_nv.nv_renderer_arch(0xA04) == "sm_120"
  assert ops_nv.nv_qmd_sass_version(0x802) == 0x82
  assert ops_nv.nv_qmd_sass_version(0x806) == 0x86
  assert ops_nv.nv_pcas_action(nv_gpu.AMPERE_COMPUTE_A) == \
    nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
  assert ops_nv.nv_pcas_action(nv_gpu.AMPERE_COMPUTE_B) == \
    nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE


def test_qmd_snapshot_captures_launch_and_release_contract():
  dev = SimpleNamespace(iface=SimpleNamespace(compute_class=nv_gpu.AMPERE_COMPUTE_A))
  qmd = ops_nv.QMD(dev, qmd_major_version=3, sass_version=0x82, program_address_upper=0x10,
                   program_address_lower=0x12340000, program_prefetch_addr_upper_shifted=0x1,
                   program_prefetch_addr_lower_shifted=0x23456789, program_prefetch_size=3,
                   constant_buffer_addr_upper_0=0x10, constant_buffer_addr_lower_0=0x23450000,
                   constant_buffer_size_shifted4_0=0x160, cta_raster_width=2, cta_raster_height=3,
                   cta_raster_depth=4, cta_thread_dimension0=5, cta_thread_dimension1=6,
                   cta_thread_dimension2=7, release0_enable=1, release0_address_upper=0x10,
                   release0_address_lower=0x34560000, release0_payload_upper=0, release0_payload_lower=7)

  snapshot = qmd.snapshot(0x1020002000)

  assert snapshot["address"] == 0x1020002000
  assert snapshot["size"] == 0x100
  assert len(snapshot["raw"]) == 0x200
  assert snapshot["major_version"] == 3
  assert snapshot["sass_version"] == 0x82
  assert snapshot["program_address"] == 0x1012340000
  assert snapshot["program_prefetch_address"] == 0x12345678900
  assert snapshot["program_prefetch_size"] == 3
  assert snapshot["constant_buffer0_address"] == 0x1023450000
  assert snapshot["constant_buffer0_size_shifted4"] == 0x160
  assert snapshot["grid"] == [2, 3, 4]
  assert snapshot["cta_threads"] == [5, 6, 7]
  assert snapshot["releases"] == [{"enable": 1, "address": 0x1034560000, "payload": 7},
                                  {"enable": 0, "address": 0, "payload": 0}]

def test_gr_context_buffer_snapshot_preserves_binding_contract():
  buffers = [
    SimpleNamespace(bufferType=0, size=0x160000, alignment=0x1000, bufferHandle=0x1234, physAddr=0x56780000, aperture=2,
                    pageSize=0x1000, pageCount=0x160, bIsContigous=1, bGlobalBuffer=0, bLocalBuffer=1, bDeviceDescendant=0),
    SimpleNamespace(bufferType=10, size=0x80000, alignment=0x20000, bufferHandle=0x2345, physAddr=0x67890000, aperture=2,
                    pageSize=0x1000, pageCount=0x80, bIsContigous=1, bGlobalBuffer=1, bLocalBuffer=0, bDeviceDescendant=1),
  ]

  assert ops_nv.nv_gr_ctx_buffer_snapshot(SimpleNamespace(bufferCount=2, ctxBufferInfo=buffers)) == [
    {"id": 0, "size": 0x160000, "alignment": 0x1000, "handle": 0x1234, "phys_addr": 0x56780000, "aperture": 2,
     "page_size": 0x1000, "page_count": 0x160, "contiguous": 1, "global": 0, "local": 1, "device_descendant": 0},
    {"id": 10, "size": 0x80000, "alignment": 0x20000, "handle": 0x2345, "phys_addr": 0x67890000, "aperture": 2,
     "page_size": 0x1000, "page_count": 0x80, "contiguous": 1, "global": 1, "local": 0, "device_descendant": 1},
  ]

def test_ga100_heap_scales_with_framebuffer_size():
  config = get_nv_chip_config(0x17, 0x00)
  def calculate(fb_size):
    return gsp_fw_heap_size(fb_size, config.fw_heap_os_size, config.fw_heap_min_mb, config.fw_heap_max_mb)

  assert calculate(8 << 30) == 105 << 20
  assert calculate(64 << 30) == 110 << 20
  assert calculate(80 << 30) == 112 << 20


def test_ga100_gsp_userd_layout_matches_inherited_openrm_hal():
  assert ga100_gsp_userd_layout(32) == (0x200, 0x200, 2)

@pytest.mark.parametrize(("chip_name", "unified"), (("GA100", True), ("GA102", False)))
def test_nvd_ga100_unifies_compute_and_dma_submission(chip_name, unified):
  dev = ops_nv.NVDevice.__new__(ops_nv.NVDevice)
  dev.gpfifo_area, dev.channel_group, dev.debug_channel = object(), 0xCF00000C, 0xCF00000E
  dev.copy_on_compute_queue = unified
  compute_fifo, dma_fifo, events = object(), object(), []
  dev.iface = SimpleNamespace(dma_class=nv_gpu.AMPERE_DMA_COPY_B,
                              rm_alloc=lambda parent, clss: events.append(("alloc", parent, clss)))
  def new_fifo(_area, ctxshare, channel_group, *, offset, entries, compute):
    events.append(("fifo", ctxshare, channel_group, offset, entries, compute))
    return compute_fifo if compute else dma_fifo
  dev._new_gpu_fifo = new_fifo

  ops_nv.NVDevice._setup_compute_and_dma_gpfifos(dev, 0xCF00000D)

  assert dev.compute_gpfifo is compute_fifo
  assert (dev.dma_gpfifo is compute_fifo) is unified
  if unified:
    assert events == [("fifo", 0xCF00000D, 0xCF00000C, 0, 0x10000, True),
                      ("alloc", 0xCF00000E, nv_gpu.AMPERE_DMA_COPY_B)]
  else:
    assert events == [("fifo", 0xCF00000D, 0xCF00000C, 0, 0x10000, True),
                      ("fifo", 0xCF00000D, 0xCF00000C, 0x100000, 0x10000, False)]

def test_unified_ga100_graph_copy_never_instantiates_copy_queue():
  from tinygrad.runtime.graph.hcq import HCQGraph

  class FakeDevice:
    copy_on_compute_queue, peer_group = True, "NV"
    def hw_copy_queue_t(self, *, queue_idx):
      raise AssertionError(f"copy queue {queue_idx} must not be instantiated")

  dev, compute_queue = FakeDevice(), object()
  graph = HCQGraph.__new__(HCQGraph)
  graph.comp_queues, graph.copy_queues = {dev: compute_queue}, {}
  graph.kick_signals, graph.kickoff_var = {}, object()
  graph.devices, graph.kickoff_value, graph.kernargs_bufs = [], 0, {}

  assert graph._copy_queue(dev, 0) is compute_queue
  assert graph.copy_queues == {}

def test_unified_ga100_setup_isolates_each_engine_stage(monkeypatch):
  events = []

  class FakeComputeQueue:
    def setup(self, **kwargs):
      events.append(("setup", kwargs))
      return self

    def signal(self, _signal, value):
      events.append(("signal", value))
      return self

    def submit(self, dev):
      events.append(("submit", dev.ga100_setup_stage))
      return self

  class UnexpectedCopyQueue:
    def __init__(self):
      raise AssertionError("unified setup must not instantiate a copy queue")

  monkeypatch.setattr(ops_nv, "NVComputeQueue", FakeComputeQueue)
  monkeypatch.setattr(ops_nv, "NVCopyQueue", UnexpectedCopyQueue)
  dev = ops_nv.NVDevice.__new__(ops_nv.NVDevice)
  dev.copy_on_compute_queue = True
  dev.iface = SimpleNamespace(compute_class=nv_gpu.AMPERE_COMPUTE_A, dma_class=nv_gpu.AMPERE_DMA_COPY_A)
  dev.timeline_signal, dev.timeline_value = object(), 1
  dev.synchronize = lambda: events.append(("sync", dev.ga100_setup_stage, dev.timeline_value - 1))

  ops_nv.NVDevice._setup_gpfifos(dev)

  assert dev.ga100_setup_stage == "complete"
  assert events == [
    ("signal", 1), ("submit", "channel_semaphore"), ("sync", "channel_semaphore", 1),
    ("setup", {"compute_class": nv_gpu.AMPERE_COMPUTE_A, "shared_mem_window": 0x729400000000,
               "local_mem_window": 0x729300000000}),
    ("signal", 2), ("submit", "compute_object"), ("sync", "compute_object", 2),
    ("setup", {"copy_class": nv_gpu.AMPERE_DMA_COPY_A}),
    ("signal", 3), ("submit", "copy_object"), ("sync", "copy_object", 3),
  ]


def test_context_repromotion_reuses_buffers_without_allocating():
  mapping = SimpleNamespace(va_addr=0x100200000, paddrs=[(0x80000000, 0x20000)])
  controls = []
  gsp = SimpleNamespace(
    nvdev=SimpleNamespace(mm=SimpleNamespace(valloc=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected allocation")))),
    rpc_rm_control=lambda **kwargs: controls.append(kwargs),
  )

  result = NV_GSP.promote_ctx(gsp, 0xC1000000, 0xCF000009, 0xCF00000E,
    {0: GRBufDesc(size=0x20000, virt=True, phys=True)}, bufs={0: mapping}, phys=False)

  assert result == {0: mapping}
  assert len(controls) == 1
  entry = controls[0]["params"].promoteEntry[0]
  assert (entry.gpuPhysAddr, entry.gpuVirtAddr, entry.bInitialize) == (0, mapping.va_addr, 0)


@pytest.mark.parametrize(("chip_name", "userd_size", "userd_cache"), (("GA100", 0x200, 2), ("GA102", 0x400, 0)))
def test_user_gpfifo_uses_chip_userd_contract(chip_name, userd_size, userd_cache):
  gsp = SimpleNamespace(
    gpfifo_class=nv_gpu.AMPERE_CHANNEL_GPFIFO_A,
    compute_class=nv_gpu.AMPERE_COMPUTE_A,
    priv_root=0xC1E00004,
    handle_gen=iter([0xCF00000E]),
    cmd_q=SimpleNamespace(send_rpc=lambda *_args: None),
    stat_q=SimpleNamespace(wait_resp=lambda *_args: b""),
    nvdev=SimpleNamespace(
      chip_name=chip_name,
      mm=SimpleNamespace(valloc=lambda *_args, **_kwargs: SimpleNamespace(paddrs=[(0x01000000, 0x1000)])),
      _alloc_boot_mem=lambda *_args, **_kwargs: (None, 0x02000000, None),
    ),
  )
  params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
    hObjectError=1,
    hUserdMemory=(ctypes.c_uint32 * 8)(0x03000000),
    userdOffset=(ctypes.c_uint64 * 8)(0x80000),
    errorNotifierMem=nv_gpu.NV_MEMORY_DESC_PARAMS(base=0x123450000, size=0xECC, addressSpace=1, cacheAttrib=0),
  )

  result = NV_GSP.rpc_rm_alloc(gsp, 0xCF00000C, nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params, client=0xC1000000)

  assert result == 0xCF00000E
  assert params.userdMem.base == 0x03080000
  assert params.userdMem.size == userd_size
  assert params.userdMem.cacheAttrib == userd_cache
  assert params.errorNotifierMem.base == 0x123450000
  assert params.errorNotifierMem.size == 0xECC
  assert params.errorNotifierMem.addressSpace == 1


@pytest.mark.parametrize("chip_name", ("GA100", "GA102"))
def test_user_compute_allocation_preserves_chip_promotion_contract(chip_name):
  promotions, user_allocs = [], {0: object(), 1: object(), 2: object()}
  golden_allocs = {0: object(), 2: object(), 3: object(), 9: object(), 10: object(), 11: object()}

  def promote_ctx(*args, **kwargs):
    promotions.append((args, kwargs))
    return user_allocs if kwargs == {"virt": False} else {}

  gsp = SimpleNamespace(
    gpfifo_class=nv_gpu.AMPERE_CHANNEL_GPFIFO_A,
    compute_class=nv_gpu.AMPERE_COMPUTE_A,
    priv_root=0xC1E00004,
    handle_gen=iter([0xCF00000F]),
    cmd_q=SimpleNamespace(send_rpc=lambda *_args: None),
    stat_q=SimpleNamespace(wait_resp=lambda *_args: b""),
    subdevice=0xCF000009,
    nvdev=SimpleNamespace(chip_name=chip_name),
    grctx_bufs={
      0: GRBufDesc(size=0x160000, virt=True, phys=True),
      1: GRBufDesc(size=0x5000, virt=True, phys=True),
      2: GRBufDesc(size=0x5000, virt=True, phys=True),
      3: GRBufDesc(size=0x20000, virt=True, phys=False),
      10: GRBufDesc(size=0x80000, virt=False, phys=True),
      9: GRBufDesc(size=0x10000, virt=True, phys=True),
      11: GRBufDesc(size=0x80000, virt=True, phys=True),
    },
    grctx_buf_allocs=golden_allocs,
    promote_ctx=promote_ctx,
  )

  result = NV_GSP.rpc_rm_alloc(gsp, 0xCF00000E, nv_gpu.AMPERE_COMPUTE_A, None, client=0xC1000000)

  assert result == 0xCF00000F
  assert len(promotions) == 2
  assert promotions[0][0][:3] == (0xC1000000, 0xCF000009, 0xCF00000E)
  assert set(promotions[0][0][3]) == {0, 1, 2}
  assert promotions[0][1] == {"virt": False}
  if chip_name == "GA100":
    assert set(promotions[1][0][3]) == {0, 1, 2, 9, 10, 11}
    assert promotions[1][0][4] == {k:(golden_allocs | user_allocs)[k] for k in {0, 1, 2, 9, 10, 11}}
    assert promotions[1][1] == {"virt": True, "phys": False}
  else:
    assert set(promotions[1][0][3]) == {0, 1, 2}
    assert promotions[1][0][4] == user_allocs
    assert promotions[1][1] == {"phys": False}


def test_ga100_flcn_skips_fwsec_frts():
  flcn = NV_FLCN_GA100.__new__(NV_FLCN_GA100)
  calls = []
  flcn.init_regs = lambda: calls.append("regs")
  flcn.prep_ucode = lambda: calls.append("fwsec-frts")
  flcn.prep_booter = lambda: calls.append("booter")

  flcn.init_sw()
  flcn.init_frts()

  assert calls == ["regs", "booter"]


def test_pci_iface_accepts_ga100_prefix():
  mask, prefixes = ops_nv.NV_PCI_DEVICES[0]
  assert mask == 0xFF00
  assert 0x2000 in prefixes
