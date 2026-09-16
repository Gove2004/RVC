"""io 层测试 — WAV 读写往返 + ffmpeg 异常契约（S5 新增覆盖）。

异常契约基线（重写必须保持）：
- resolve_ffmpeg / load_audio：ffmpeg 缺失 → FileNotFoundError（环境问题），
  解码失败 → AudioLoadError（数据问题）——二元性是刻意设计；
- write_wav：非法 subtype → ValueError；PCM_16 越界样本 clip（不报错）；
- read_wav_info：非 WAV/缺块 → ValueError。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvc.core.errors import AudioLoadError
from rvc.io.audio_file import load_audio, resolve_ffmpeg
from rvc.io.wav_file import read_audio_info, read_wav_info, write_wav

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_INPUT = PROJECT_ROOT / "tests" / "golden" / "input_48k.wav"


class TestResolveFfmpeg(unittest.TestCase):
    def test_missing_explicit_path_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            resolve_ffmpeg("Z:/definitely/not/here/ffmpeg.exe")

    def test_default_resolution(self):
        # 本机 assets/ffmpeg/ffmpeg.exe 应存在；不存在则整个用例组跳过
        exe = resolve_ffmpeg()
        assert exe.exists()


@unittest.skipUnless(GOLDEN_INPUT.exists(), "golden input wav 缺失")
class TestLoadAudio(unittest.TestCase):
    def test_missing_ffmpeg_exe_beats_decode(self):
        # 显式传一个不存在的 ffmpeg → FileNotFoundError（不是 AudioLoadError）
        with self.assertRaises(FileNotFoundError):
            load_audio(GOLDEN_INPUT, 16000, ffmpeg_exe="Z:/nope/ffmpeg.exe")

    def test_decode_failure_raises_audio_load_error(self):
        # 真 ffmpeg + 坏文件 → AudioLoadError（数据问题）
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.wav"
            bad.write_bytes(b"this is not audio data at all")
            with self.assertRaises(AudioLoadError):
                load_audio(bad, 16000)

    def test_decode_roundtrip_mono(self):
        wav, sr = load_audio(GOLDEN_INPUT, 48000)
        assert sr == 48000
        assert wav.ndim == 1
        assert np.isfinite(wav).all()
        assert wav.size > 0

    def test_decode_resample_changes_length(self):
        wav48, _ = load_audio(GOLDEN_INPUT, 48000)
        wav16, _ = load_audio(GOLDEN_INPUT, 16000)
        assert abs(wav16.size / wav48.size - 16000 / 48000) < 0.02

    def test_stereo_shape(self):
        wav, sr = load_audio(GOLDEN_INPUT, 48000, mono=False)
        assert sr == 48000
        assert wav.shape[0] == 2


class TestWavFile(unittest.TestCase):
    def test_float_roundtrip(self):
        data = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.wav"
            write_wav(path, data, 48000, subtype="FLOAT")
            info = read_wav_info(path)
            assert info["samplerate"] == 48000
            assert info["channels"] == 1
            assert info["frames"] == 1000
            assert info["bits_per_sample"] == 32
            assert abs(info["duration"] - 1000 / 48000) < 1e-9

    def test_pcm16_clips_out_of_range(self):
        data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.wav"
            write_wav(path, data, 16000, subtype="PCM_16")  # 不应报错
            info = read_wav_info(path)
            assert info["bits_per_sample"] == 16
            assert info["frames"] == 4

    def test_invalid_subtype_raises_valueerror(self):
        data = np.zeros(10, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                write_wav(Path(td) / "t.wav", data, 16000, subtype="PCM_24")

    def test_read_garbage_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.wav"
            bad.write_bytes(b"RIFFxxxxWAVEjunkjunkjunk")
            with self.assertRaises(ValueError):
                read_wav_info(bad)


class TestReadAudioInfoFfmpegParam(unittest.TestCase):
    def test_non_wav_uses_explicit_ffmpeg_path(self):
        # S9/D4：read_audio_info 的 ffmpeg_path 显式传参也必须真实生效——
        # 传不存在的路径 → FileNotFoundError（若参数被吞，会静默走默认路径）
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "fake.mp3"
            fake.write_bytes(b"\x00" * 64)
            with self.assertRaises(FileNotFoundError):
                read_audio_info(fake, ffmpeg_path="Z:/nope/ffmpeg.exe")


class TestTrainPreprocessFfmpegParam(unittest.TestCase):
    def test_preprocessor_threads_ffmpeg_exe(self):
        # S9/D4：PreProcessor 的 ffmpeg_exe 显式传参必须贯穿到 load_audio——
        # 传不存在的路径 → FileNotFoundError（若被吞，会用默认路径成功解码）
        from rvc.train.preprocess import PreProcessor

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            data = np.zeros(4800, dtype=np.float32)
            write_wav(src / "a.wav", data, 48000, subtype="FLOAT")
            exp = Path(td) / "exp"
            processor = PreProcessor(str(src), str(exp), 48000, per=3.7,
                                     ffmpeg_exe="Z:/nope/ffmpeg.exe")
            with self.assertRaises(FileNotFoundError):
                processor.run()


if __name__ == "__main__":
    unittest.main()
