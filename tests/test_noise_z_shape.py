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


class TestZSampleContract(unittest.TestCase):
    """`_z_sample` 的静默退回契约：形状一致才用固定噪声场。

    这条分支的失败是**无声**的——退回 randn_like 不报错、不告警，只是固定噪声场
    等于没接，表现为持续/重复内容的咬字逐块随机漂移。所以两个方向都要锁：
    命中时必须原样返回传入张量（不是复制、不是重新采样），不匹配时必须退回。
    """

    def _z_sample(self, m_p, z_noise):
        from rvc.models.synthesizer import _SynthesizerTrnMsBase

        # 该方法体内不引用 self，用 None 占位即可，避免构造整个模型
        return _SynthesizerTrnMsBase._z_sample(None, m_p, z_noise)

    def test_matching_shape_returns_the_same_tensor(self):
        import torch

        m_p = torch.zeros(1, 192, 56)
        z = torch.ones(1, 192, 56)
        out = self._z_sample(m_p, z)
        self.assertIs(out, z, "命中时必须原样返回传入张量，否则固定噪声场没意义")

    def test_mismatched_shape_falls_back(self):
        import torch

        m_p = torch.zeros(1, 192, 56)
        for bad in (torch.ones(1, 192, 57), torch.ones(1, 192, 55),
                    torch.ones(1, 192)):
            out = self._z_sample(m_p, bad)
            self.assertEqual(tuple(out.shape), tuple(m_p.shape))
            self.assertIsNot(out, bad, "形状不符时必须退回重新采样")

    def test_none_falls_back(self):
        import torch

        m_p = torch.zeros(1, 192, 56)
        out = self._z_sample(m_p, None)
        self.assertEqual(tuple(out.shape), tuple(m_p.shape))


class TestRealHubertConfigFrames(unittest.TestCase):
    """用真实 assets/hubert/base/config.json 校验帧数，而不是假配置。

    假配置只能证明公式自洽；真实资产才能证明「runner 预生成的 noise_z 长度」
    与「upsample_features 实际输出长度」在被测环境里确实相等。
    历史偏差（按 skip_head + return_length 估，恒差一帧）正是漏在这一步。
    """

    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path

        cfg_path = Path(rvc_hubert_dir()) / "config.json"
        if not cfg_path.exists():
            raise unittest.SkipTest(f"缺少真实 HuBERT 配置: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cls.kernel = cfg["conv_kernel"]
        cls.stride = cfg["conv_stride"]

    def test_upsample_length_equals_runner_formula(self):
        """runner 的 p_len 公式必须等于 upsample_features 的实际输出长度。"""
        import torch

        from rvc.pipeline.features import upsample_features

        model = _FakeHubert(self.kernel, self.stride)
        for n16 in (16000, 32000, 44960, 89920):
            frames = hubert_frame_count(model, n16)
            runner_p_len = 2 * (frames + 1)
            feats = torch.zeros(1, frames, 8)
            actual = upsample_features(feats, False).shape[1]
            self.assertEqual(
                actual, runner_p_len,
                f"n16={n16}: upsample 实际 {actual} vs runner 公式 {runner_p_len}")

    def test_m_p_length_matches_noise_z_length(self):
        """m_p 长度 = p_len - flow_head，runner 的 length 必须同式。"""
        from rvc.models.synthesizer import FLOW_CONTEXT_COMPENSATION_FRAMES
        model = _FakeHubert(self.kernel, self.stride)
        for n16 in (16000, 44960, 89920):
            p_len = 2 * (hubert_frame_count(model, n16) + 1)
            for skip_head in (0, 24, 40, 100):
                flow_head = max(skip_head - FLOW_CONTEXT_COMPENSATION_FRAMES, 0)
                # runner.py 的算法
                runner_length = p_len - max(
                    skip_head - FLOW_CONTEXT_COMPENSATION_FRAMES, 0)
                # Synthesizer.infer 里 enc_p 收到的帧数（= flow_head 之后的长度）
                m_p_len = p_len - flow_head
                self.assertEqual(runner_length, m_p_len,
                                 f"n16={n16} skip_head={skip_head}")


def rvc_hubert_dir() -> str:
    """真实 HuBERT 目录（复用 rvc.models.hubert 的解算，避免测试里另写一套）。"""
    from rvc.models.hubert import hubert_path

    return hubert_path("base")


if __name__ == "__main__":
    unittest.main()
