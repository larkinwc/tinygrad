import struct

import pytest

from tinygrad.runtime import ops_nv
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.support.nv.ip import NV_FLCN_GA100, gsp_fw_heap_size, parse_riscv_ucode_desc
from tinygrad.runtime.support.nv.nvdev import get_nv_chip_config, require_ga100_vram_size


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


def test_ga100_vram_preflight_is_fail_closed():
  ga100, ga102 = get_nv_chip_config(0x17, 0x00), get_nv_chip_config(0x17, 0x02)

  require_ga100_vram_size(ga100, 8192, 8192)
  require_ga100_vram_size(ga102, 0, 0)
  with pytest.raises(RuntimeError, match="requires NV_EXPECTED_VRAM_MIB"):
    require_ga100_vram_size(ga100, 8192, 0)
  with pytest.raises(RuntimeError, match="expected 65536 MiB, scratch reports 8192 MiB"):
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
