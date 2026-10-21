"""rvc.dsp.hz_midi — Hz↔MIDI 换算基础不变量。"""
import unittest

import torch

from rvc.dsp.hz_midi import hz_to_midi, midi_to_hz


class TestHzMidi(unittest.TestCase):
    def test_a4_is_69(self):
        self.assertAlmostEqual(float(hz_to_midi(torch.tensor(440.0))), 69.0, places=5)

    def test_octave_is_12_semitones(self):
        m1 = float(hz_to_midi(torch.tensor(220.0)))
        m2 = float(hz_to_midi(torch.tensor(440.0)))
        self.assertAlmostEqual(m2 - m1, 12.0, places=5)

    def test_round_trip(self):
        freqs = torch.tensor([30.0, 126.3, 440.0, 720.0, 1000.0])
        back = midi_to_hz(hz_to_midi(freqs))
        self.assertTrue(torch.allclose(back, freqs, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
