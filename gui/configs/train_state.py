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
        }
