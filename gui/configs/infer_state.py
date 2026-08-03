"""推理 GUI 状态对象。"""
from dataclasses import dataclass
from typing import Any


@dataclass
class InferGuiState:
    block_time: float
    crossfade_time: float
    extra_time: float
    protect: float
    f0method: str
    sr_mode: str
    eq_enabled: bool
    eq_low: float
    eq_mid: float
    eq_high: float
    reverb: float
    reverb_enable: bool
    rms_mix: float
    nr_enable: bool
    nr_strength: float
    bgm_enable: bool
    bgm_path: str
    bgm_vol: float
    preset: str
    hostapi: str
    input_device: str
    output_device: str
    output2_device: str
    active_model: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferGuiState":
        return cls(
            block_time=float(data.get("bl", 0.25)),
            crossfade_time=float(data.get("cf", 0.05)),
            extra_time=float(data.get("ex", 2.5)),
            protect=float(data.get("protect", 0.25)),
            f0method=str(data.get("f0", "fcpe")),
            sr_mode=str(data.get("sr_mode", "model")),
            eq_enabled=bool(data.get("eq_en", False)),
            eq_low=float(data.get("eq_lo", 0.0)),
            eq_mid=float(data.get("eq_mi", 0.0)),
            eq_high=float(data.get("eq_hi", 0.0)),
            reverb=float(data.get("rev", 0.0)),
            reverb_enable=bool(data.get("rev_en", True)),
            rms_mix=float(data.get("rms", 0.0)),
            nr_enable=bool(data.get("nr_en", False)),
            nr_strength=float(data.get("nr_str", 0.5)),
            bgm_enable=bool(data.get("bgm_en", False)),
            bgm_path=str(data.get("bgm_path", "")),
            bgm_vol=float(data.get("bgm_vol", 0.5)),
            preset=str(data.get("preset", "默认")),
            hostapi=str(data.get("ha", "")),
            input_device=str(data.get("in_dev", "")),
            output_device=str(data.get("out_dev", "")),
            output2_device=str(data.get("out2_dev", "")),
            active_model=str(data.get("active_model", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bl": self.block_time,
            "cf": self.crossfade_time,
            "ex": self.extra_time,
            "protect": self.protect,
            "f0": self.f0method,
            "sr_mode": self.sr_mode,
            "eq_en": self.eq_enabled,
            "eq_lo": self.eq_low,
            "eq_mi": self.eq_mid,
            "eq_hi": self.eq_high,
            "rev": self.reverb,
            "rev_en": self.reverb_enable,
            "rms": self.rms_mix,
            "nr_en": self.nr_enable,
            "nr_str": self.nr_strength,
            "bgm_en": self.bgm_enable,
            "bgm_path": self.bgm_path,
            "bgm_vol": self.bgm_vol,
            "preset": self.preset,
            "ha": self.hostapi,
            "in_dev": self.input_device,
            "out_dev": self.output_device,
            "out2_dev": self.output2_device,
            "active_model": self.active_model,
        }
