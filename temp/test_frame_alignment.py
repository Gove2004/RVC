"""测试 HuBERT 和 F0 输出帧数是否对齐。"""
import torch
import sys
sys.path.insert(0, '.')

from rvc.pipeline.features import extract_hubert_features, upsample_features
from rvc.pipeline.pitch.tracker import extract_f0_for_block
from rvc.core.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE

# 模拟一段输入
sr = 16000
duration = 5.0  # 5 秒
n_samples = int(sr * duration)
input_wav = torch.randn(n_samples, device='cuda')

p_len = input_wav.shape[0] // HUBERT_FRAME_SIZE

print(f"输入样本数: {input_wav.shape[0]}")
print(f"p_len (100fps): {p_len}")

# 模拟 HuBERT 输出（假设有 258 帧，50fps）
# 实际应该从模型提取，但我们先估算
hubert_est_frames = input_wav.shape[0] // 320
print(f"HuBERT 估算帧数 (50fps): {hubert_est_frames}")

# 模拟 upsample 后的长度
feats_fake = torch.randn(1, hubert_est_frames, 768, device='cuda')
feats_up = upsample_features(feats_fake, p_len, is_half=True)
print(f"upsample 后特征长度: {feats_up.shape[1]}")

# 模拟 F0 输出
f0_fake = torch.randn(p_len + 3, device='cuda')  # 假设多 3 帧
conf_fake = torch.randn(p_len + 3, device='cuda')

# 对齐
n = f0_fake.shape[0]
if n >= p_len:
    f0_aligned = f0_fake[:p_len]
else:
    f0_aligned = torch.nn.functional.pad(f0_fake, (0, p_len - n))

print(f"F0 原始长度: {n}")
print(f"F0 对齐后长度: {f0_aligned.shape[0]}")

print("\n对齐后两者应该都是", p_len)
