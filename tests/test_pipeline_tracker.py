"""pipeline pitch tracker 缓存帧对齐契约测试。

锁定 RMVPE 边缘帧丢弃语义（RMVPE_F0_*_MARGIN_FRAMES）：
cache[N-4:] 接收 f0_raw[3:N-1]，头部 3 帧 + 尾部 1 帧被丢弃。
数值行为与旧实现 f0_raw[3:-1] / write_start = 4 - len 完全一致。
"""
import unittest

import torch

from rvc.pipeline.pitch.tracker import (
    RMVPE_F0_DISCARDED_FRAMES,
    RMVPE_F0_HEAD_MARGIN_FRAMES,
    create_pitch_cache,
    realtime_f0_window,
    update_realtime_pitch_cache_raw,
)


class _FakeExtractor:
    """返回可预测 f0/conf 的桩提取器（不加载真实模型）。"""

    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def extract_raw(self, window, sr):
        f0 = torch.arange(self.num_frames, dtype=torch.float32) * 10.0
        confidence = torch.full((self.num_frames,), 0.5)
        return f0, confidence


def _update(cache_pitch, cache_pitchf, cache_conf, block_frame_16k, num_frames=40):
    return update_realtime_pitch_cache_raw(
        input_wav=torch.zeros(16000),
        block_frame_16k=block_frame_16k,
        p_len=64,
        method="fcpe",
        cache_pitch=cache_pitch,
        cache_pitchf=cache_pitchf,
        cache_confidence=cache_conf,
        device="cpu",
        is_half=False,
        inference_cache=None,
        extractor=_FakeExtractor(num_frames),
    )


class TestTrackerCacheAlignment(unittest.TestCase):
    def test_edge_frames_discarded(self):
        num_frames = 40
        cache_pitch, cache_pitchf, cache_conf = create_pitch_cache("cpu")
        _update(cache_pitch, cache_pitchf, cache_conf, 160, num_frames)
        expected = torch.arange(num_frames, dtype=torch.float32) * 10.0
        # 缓存尾部 (N-4) 槽位 == f0_raw[3:N-1]
        tail_slice = cache_pitchf[RMVPE_F0_DISCARDED_FRAMES - num_frames:]
        self.assertTrue(torch.equal(tail_slice, expected[3:num_frames - 1]))
        conf_slice = cache_conf[RMVPE_F0_DISCARDED_FRAMES - num_frames:]
        self.assertTrue(torch.allclose(conf_slice, torch.full_like(conf_slice, 0.5)))

    def test_values_identical_to_legacy_slice(self):
        """与旧实现逐元素等价：f0_raw[3:-1] / write_start = 4 - len。"""
        num_frames = 40
        cache_pitch, cache_pitchf, cache_conf = create_pitch_cache("cpu")
        _update(cache_pitch, cache_pitchf, cache_conf, 160, num_frames)
        legacy = torch.zeros_like(cache_pitchf)
        legacy[4 - num_frames:] = (torch.arange(num_frames, dtype=torch.float32) * 10.0)[3:-1]
        self.assertTrue(torch.equal(cache_pitchf, legacy))

    def test_cache_left_shift(self):
        """新块到来时：旧值左移 shift 槽，尾部 (N-4) 槽被新帧覆盖。"""
        num_frames = 40
        shift = 2
        old = torch.arange(2048, dtype=torch.float32)
        cache_pitch, cache_pitchf, cache_conf = create_pitch_cache("cpu")
        cache_pitchf[:] = old
        _update(cache_pitch, cache_pitchf, cache_conf, 160 * shift)
        expected = torch.empty_like(cache_pitchf)
        expected[:-shift] = old[shift:]                      # 左移
        tail = torch.arange(3, num_frames - 1, dtype=torch.float32) * 10.0
        expected[-(num_frames - 4):] = tail                  # 尾部覆盖（新帧）
        self.assertTrue(torch.equal(cache_pitchf, expected))

    def test_output_views_match_cache_tail(self):
        cache_pitch, cache_pitchf, cache_conf = create_pitch_cache("cpu")
        f0_out, conf_out = _update(cache_pitch, cache_pitchf, cache_conf, 160)
        self.assertTrue(torch.equal(f0_out[0], cache_pitchf[-64:]))
        self.assertTrue(torch.equal(conf_out[0], cache_conf[-64:]))


class TestWindowLength(unittest.TestCase):
    def test_fcpe_window(self):
        self.assertEqual(realtime_f0_window(160, "fcpe"), 160 + 800)

    def test_rmvpe_window_aligned_to_5120(self):
        window = realtime_f0_window(160, "rmvpe")
        self.assertEqual((window + 160) % 5120, 0)


if __name__ == "__main__":
    unittest.main()
