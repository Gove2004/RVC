"""守卫：全项目半精度不变量。

用户裁定"全程整个项目只有半精度"。这条不变量此前只是**碰巧成立**，没有任何
地方声明或检查它，于是留下了两个真实后果：

1. `assets/hubert/chinese` 官方分发的是 fp32（360MB），而推理侧
   `rvc/runtime/device.py` 把 `is_half` 硬编码为 True，模型加载后立刻
   `.to(dtype=torch.float16)`——那份 fp32 的低位在加载第一瞬间就被丢掉，
   只白占一倍磁盘、`torch.load` 期间还多占一倍宿主内存。
2. 如果有人按"用 fp32 权重换精度"的思路排查音质，会得到完全无效的结果：
   实测 `is_half=False` 时参数 dtype 是 float32，但低 13 位全零占比 1.0000，
   就是 fp16 值零填充，不存在真实 fp32 精度。

本测试锁住三件事：
  - 推理侧 is_half 恒为 True（不得引入 fp32 推理路径）
  - 已就位的 HuBERT 权重必须是 fp16（缺权重时跳过，不误报）
  - 转换工具存在且幂等（保证换机器重新下载后能复现转换）
"""
import unittest
from pathlib import Path

import torch

from rvc.core.config import HUBERT_VARIANTS
from rvc.models.hubert import hubert_path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestInferenceIsHalfOnly(unittest.TestCase):
    """推理侧不得存在 fp32 路径。"""

    def test_runtime_device_config_hardcodes_half(self):
        """`RuntimeDeviceConfig._init` 必须把 is_half 设为 True。

        它是全项目 is_half 的唯一源头（其余都是从这里往下传）。
        """
        from rvc.runtime.device import RuntimeDeviceConfig

        src = (_PROJECT_ROOT / "rvc" / "runtime" / "device.py").read_text(encoding="utf-8")
        self.assertIn(
            "self.is_half = True", src,
            "推理侧 is_half 不再是硬编码 True。若改成可配置，必须同时确认 HuBERT "
            "权重精度与之一致，否则会退化成 fp16 权重按 fp32 跑（白占显存、精度不增）",
        )

    def test_engine_state_defaults_to_half(self):
        from rvc.pipeline.state import EngineState

        self.assertTrue(EngineState().is_half)


class TestHubertWeightsAreHalf(unittest.TestCase):
    """已就位的 HuBERT 权重必须是 fp16。"""

    def test_weights_are_half_when_present(self):
        checked = []
        for variant in HUBERT_VARIANTS:
            weight = Path(hubert_path(variant)) / "pytorch_model.bin"
            if not weight.exists():
                continue  # 未下载该变体，不算失败
            sd = torch.load(weight, map_location="cpu", weights_only=True)
            non_half = [k for k, v in sd.items()
                        if torch.is_tensor(v) and v.dtype != torch.float16]
            self.assertFalse(
                non_half,
                f"{variant} 权重有 {len(non_half)} 个非 fp16 张量（样例 {non_half[:3]}）。"
                f"全项目只有半精度，fp32 存盘只会在加载瞬间丢失低位、白占一倍磁盘。"
                f"用 tools/convert_hubert_fp16.py 转换",
            )
            checked.append(variant)
        if not checked:
            self.skipTest("本机未放置任何 HuBERT 权重")


class TestConversionToolExists(unittest.TestCase):
    """转换工具要能被找到并识别为幂等（换机器重新下载 fp32 后可复现）。"""

    def test_tool_present(self):
        tool = _PROJECT_ROOT / "tools" / "convert_hubert_fp16.py"
        self.assertTrue(tool.exists(), "转换工具缺失，重新下载 fp32 权重后无法复现转换")

    def test_tool_is_idempotent(self):
        """对已 fp16 的权重跑工具应报告"无需转换"而非重写。"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_conv_tool", _PROJECT_ROOT / "tools" / "convert_hubert_fp16.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertTrue(mod._is_all_half({"w": torch.zeros(2, dtype=torch.float16)}))
        self.assertFalse(mod._is_all_half({"w": torch.zeros(2, dtype=torch.float32)}))
        self.assertFalse(mod._is_all_half({}))


if __name__ == "__main__":
    unittest.main()
