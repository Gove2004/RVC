"""Hz ↔ MIDI 半音互转（A4 = 440Hz = MIDI 69）。

自动适配 torch/numpy 输入；推理（GPU/torch）与训练（CPU/numpy）共用，
避免两处独立实现导致口径漂移。
"""
import numpy as np
import torch

# MIDI 参考：A4 = 440Hz = MIDI 69
MIDI_REF_FREQ = 440.0
MIDI_REF_NOTE = 69


def hz_to_midi(freq, xp=None):
    """Hz → MIDI 半音（A4=440Hz = MIDI 69）。自动适配 torch/numpy。"""
    if xp is None:
        xp = torch if torch.is_tensor(freq) else np
    if xp is torch and not torch.is_tensor(freq):
        freq = torch.tensor(freq, dtype=torch.float32)
    return 12.0 * xp.log2(freq / MIDI_REF_FREQ) + MIDI_REF_NOTE


def midi_to_hz(midi, xp=None):
    """MIDI 半音 → Hz。自动适配 torch/numpy。"""
    if xp is None:
        xp = torch if torch.is_tensor(midi) else np
    if xp is torch and not torch.is_tensor(midi):
        midi = torch.tensor(midi, dtype=torch.float32)
    return MIDI_REF_FREQ * (2.0 ** ((midi - MIDI_REF_NOTE) / 12.0))
