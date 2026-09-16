"""离线转换端到端测试（重建 S2 前被删的死金标准覆盖）。

用真实模型 + golden 短 wav 走 VoiceEngine.process_file 全链路：
ffmpeg 解码 → 逐块实时同源推理 → 峰值归一化 → FLOAT WAV 落盘。
断言输出结构契约（采样率/时长/格式/非静音），不做逐位金标准对比
（跨 torch/GPU 版本脆弱，结构契约已足以锁定链路连通与行为量级）。

现状契约：离线与实时共用同一 engine/runner（用户裁定），故本测试
同时覆盖 process_block 的离线路径。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rvc.core.config import OfflineParams
from rvc.io.audio_file import load_audio
from rvc.io.wav_file import read_wav_info
from rvc.pipeline.cache import default_inference_cache
from rvc.runtime.paths import MODELS_DIR
from rvc.streaming.engine import VoiceEngine

INPUT_WAV = Path(__file__).parent / "golden" / "input_48k.wav"


def _pick_smallest_model() -> Path | None:
    if not MODELS_DIR.is_dir():
        return None
    models = sorted(MODELS_DIR.glob("*.pth"), key=lambda p: p.stat().st_size)
    return models[0] if models else None


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
@unittest.skipUnless(INPUT_WAV.exists(), "缺少 golden 输入 wav")
class TestOfflineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = _pick_smallest_model()
        if cls.model_path is None:
            raise unittest.SkipTest("assets/models/ 无可用模型")
        # 模型加载最重（HuBERT+RMVPE+合成器），类内共享一个引擎；
        # process_file 每次自建 runner，实例复用安全（现状契约）
        cls._engine = VoiceEngine(OfflineParams(), inference_cache=default_inference_cache)
        cls.engine_sr = cls._engine.load_model(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._engine.stop()

    def _convert(self, tmpdir: Path) -> Path:
        """跑一次完整离线转换，返回输出文件路径。"""
        task = OfflineParams(input_path=str(INPUT_WAV))
        task.output_path = str(tmpdir / "out.wav")
        self._engine.process_file(task)
        return Path(task.output_path)

    def test_process_file_writes_valid_float_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = self._convert(Path(tmp))
            self.assertTrue(out_path.exists())

            info = read_wav_info(out_path)
            # 输出采样率 == 模型目标采样率；位深 32 = FLOAT
            self.assertEqual(info["samplerate"], self.engine_sr)
            self.assertEqual(info["bits_per_sample"], 32)

            # 输出时长 == 输入时长（pad 被精确裁剪；输入 48k，帧数一致）
            src_info = read_wav_info(INPUT_WAV)
            self.assertAlmostEqual(info["duration"], src_info["duration"], delta=0.05)

    def test_output_is_finite_and_non_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = self._convert(Path(tmp))
            wav, _ = load_audio(out_path, 48000)
            self.assertTrue(np.isfinite(wav).all())
            self.assertGreater(float(np.abs(wav).max()), 1e-4)   # 非静音
            self.assertLessEqual(float(np.abs(wav).max()), 1.0)  # 峰值归一化到 0.99


if __name__ == "__main__":
    unittest.main()
