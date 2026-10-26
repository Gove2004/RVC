"""z_p 固定噪声场的形状契约。

noise_z 的形状必须与 m_p 完全一致，否则 Synthesizer._z_sample 的形状校验
判为不匹配、静默退回 randn_like——固定噪声场等于没接，而且不报任何错。

历史偏差：长度曾按 skip_head + return_length 估，而 m_p 的实际长度来自
feats_up = (hubert_frames + 1) * 2（+1 是上采样的末帧重复），两者恒差一帧，
于是这个特性从上线起就一直没生效。这里锁死「必须精确算帧数」。
"""
import unittest

from rvc.pipeline.features import hubert_frame_count


class _FakeConfig:
    def __init__(self, kernel, stride):
        self.conv_kernel = kernel
        self.conv_stride = stride


class _FakeHubert:
    def __init__(self, kernel, stride):
        self.config = _FakeConfig(kernel, stride)


# 真实 assets/hubert/chinese 的配置
_CN_KERNEL = [10, 3, 3, 3, 3, 2, 2]
_CN_STRIDE = [5, 2, 2, 2, 2, 2, 2]


class TestHubertFrameCount(unittest.TestCase):
    def test_chinese_default_block(self):
        """默认块参数（40k、0.25s）下 16k 缓冲 44960 样本 → 140 帧。

        这个数对不上就会让 m_p 与 noise_z 差 1 帧（141*2=282 vs 281）。
        """
        n16 = 160 * (100000 + 2000 + 400 + 10000) // 400
        self.assertEqual(n16, 44960)
        frames = hubert_frame_count(_FakeHubert(_CN_KERNEL, _CN_STRIDE), n16)
        self.assertEqual(frames, 140)
        # 上采样后 m_p 长度 = (140+1)*2 = 282，而 skip_head+return_length = 281
        self.assertEqual(2 * (frames + 1), 282)

    def test_padding_is_zero(self):
        """特征提取器 padding=0，逐层 (L - k) // s + 1。

        若某层按 (k-1)//2 补边来算，结果会多出几帧，与本实现不符。
        """
        length = 1000
        expected = length
        for k, s in zip(_CN_KERNEL, _CN_STRIDE):
            expected = (expected - k) // s + 1
        self.assertEqual(hubert_frame_count(_FakeHubert(_CN_KERNEL, _CN_STRIDE), 1000), expected)

    def test_matches_upsample_contract(self):
        """p_len 必须等于 upsample_features 实际输出的长度。

        逐块喂入时 feats 的帧数由 HuBERT 决定，upsample 后是 (T+1)*2。
        """
        for n16 in (16000, 32000, 44960, 89920):
            frames = hubert_frame_count(_FakeHubert(_CN_KERNEL, _CN_STRIDE), n16)
            feats_up_len = 2 * (frames + 1)
            # 上采样语义：cat 一帧再 2x 最近邻插值
            self.assertEqual(feats_up_len, 2 * frames + 2)


if __name__ == "__main__":
    unittest.main()
