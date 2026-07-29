from tinygrad.runtime import ops_nv
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.nv.ip import NV_FLCN_GA100, gsp_fw_heap_size
from tinygrad.runtime.support.nv.nvdev import get_nv_chip_config


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


def test_ga100_uses_ampere_a_classes():
  ga100 = get_nv_chip_config(0x17, 0x00)
  ga102 = get_nv_chip_config(0x17, 0x02)

  assert (ga100.gpfifo_class, ga100.compute_class, ga100.dma_class) == (
    nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_A, nv_gpu.AMPERE_DMA_COPY_A)
  assert (ga102.gpfifo_class, ga102.compute_class, ga102.dma_class) == (
    nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B)


def test_ga100_heap_scales_with_framebuffer_size():
  config = get_nv_chip_config(0x17, 0x00)
  def calculate(fb_size):
    return gsp_fw_heap_size(fb_size, config.fw_heap_os_size, config.fw_heap_min_mb, config.fw_heap_max_mb)

  assert calculate(8 << 30) == 105 << 20
  assert calculate(64 << 30) == 110 << 20
  assert calculate(80 << 30) == 112 << 20


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
