"""训练 GUI 状态对象。"""
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainGuiState:
    exp_name: str
    input_dir: str
    sample_rate: str
    epochs: int
    batch_size: int
    save_every: int
    learning_rate: str
    pretrain_g: str
    pretrain_d: str
    # ── 人声提纯 Tab ──
    sep_input_dir: str = ""
    sep_output_dir: str = ""
    sep_model: str = "htdemucs"  # 主分离模型 key
    sep_out_sr: str = "44.1k"  # 输出采样率
    sep_dereverb: bool = True
    sep_karaoke: bool = False
    sep_denoise: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainGuiState":
        return cls(
            exp_name=str(data.get("exp_name", "")),
            input_dir=str(data.get("input_dir", "")),
            sample_rate=str(data.get("sr", "48k")),
            epochs=int(data.get("epochs", 200)),
            batch_size=int(data.get("batch_size", 4)),
            save_every=int(data.get("save_every", 20)),
            learning_rate=str(data.get("learning_rate", "0.0001")),
            pretrain_g=str(data.get("pretrain_g", "")),
            pretrain_d=str(data.get("pretrain_d", "")),
            sep_input_dir=str(data.get("sep_input_dir", "")),
            sep_output_dir=str(data.get("sep_output_dir", "")),
            sep_model=str(data.get("sep_model", "htdemucs")),
            sep_out_sr=str(data.get("sep_out_sr", "44.1k")),
            sep_dereverb=bool(data.get("sep_dereverb", True)),
            sep_karaoke=bool(data.get("sep_karaoke", False)),
            sep_denoise=bool(data.get("sep_denoise", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "input_dir": self.input_dir,
            "sr": self.sample_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "save_every": self.save_every,
            "learning_rate": self.learning_rate,
            "pretrain_g": self.pretrain_g,
            "pretrain_d": self.pretrain_d,
            "sep_input_dir": self.sep_input_dir,
            "sep_output_dir": self.sep_output_dir,
            "sep_model": self.sep_model,
            "sep_out_sr": self.sep_out_sr,
            "sep_dereverb": self.sep_dereverb,
            "sep_karaoke": self.sep_karaoke,
            "sep_denoise": self.sep_denoise,
        }
