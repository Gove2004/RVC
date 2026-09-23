"""训练 GUI 状态对象。

本模块承担 gui/train 的 viewmodel 职责：训练 GUI 状态的序列化/反序列化、
默认值管理、与 GUI 控件的双向绑定。训练运行时配置（TrainConfig）在 rvc.core.config。
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainGuiState:
    """训练 GUI 状态。默认值单源（D6）：字段默认即唯一缺省来源，
    from_dict 与控件初值都从这里取。"""
    exp_name: str = ""
    input_dir: str = ""
    sample_rate: str = "48k"
    epochs: int = 200
    batch_size: int = 4
    save_every: int = 20
    # UI 展示默认 "1e-4"（settings_tab 保持展示现状，D6 允许）；内部缺省写法如下
    learning_rate: str = "0.0001"
    pretrain_g: str = ""
    pretrain_d: str = ""
    # HuBERT 特征器: base / chinese（与推理侧模型卡牌选择一致，训练推理必须同一种）
    hubert: str = "chinese"

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TrainGuiState":
        defaults = cls()
        return cls(
            exp_name=str(state.get("exp_name", defaults.exp_name)),
            input_dir=str(state.get("input_dir", defaults.input_dir)),
            sample_rate=str(state.get("sr", defaults.sample_rate)),
            epochs=int(state.get("epochs", defaults.epochs)),
            batch_size=int(state.get("batch_size", defaults.batch_size)),
            save_every=int(state.get("save_every", defaults.save_every)),
            learning_rate=str(state.get("learning_rate", defaults.learning_rate)),
            pretrain_g=str(state.get("pretrain_g", defaults.pretrain_g)),
            pretrain_d=str(state.get("pretrain_d", defaults.pretrain_d)),
            hubert=str(state.get("hubert", defaults.hubert)),
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
            "hubert": self.hubert,
        }
