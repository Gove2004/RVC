"""runtime.device 设备探测契约测试。"""
import unittest
from unittest.mock import patch


class TestDeviceNoGpu(unittest.TestCase):
    """现状契约：无 CUDA 环境直接失败（不回退 CPU），但错误必须可读（中文、含判据）。"""

    def test_no_cuda_raises_readable_error(self):
        import rvc.runtime.device as device_mod

        cls = device_mod.Config
        saved = cls._instance
        cls._instance = None  # 重置单例，强制重走探测
        try:
            with patch.object(device_mod.torch.cuda, "is_available", return_value=False):
                with self.assertRaises(RuntimeError) as ctx:
                    cls()
            message = str(ctx.exception)
            self.assertIn("NVIDIA", message)
            self.assertIn("torch.cuda.is_available", message)
        finally:
            cls._instance = saved


if __name__ == "__main__":
    unittest.main()
