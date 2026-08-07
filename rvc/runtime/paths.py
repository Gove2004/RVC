"""运行时路径。"""
from pathlib import Path

CONFIG_ROOT = Path("assets/configs")
STATE_FILE = CONFIG_ROOT / "save_state.json"

# 训练采样率 → 配置文件（40k/48k）
TRAIN_CONFIG_FILES = {
    40000: CONFIG_ROOT / "40ktrain_config.json",
    48000: CONFIG_ROOT / "48ktrain_config.json",
}
DEFAULT_TRAIN_SR = 48000


def config_path() -> Path:
    return STATE_FILE


def _to_hz(sr) -> int:
    """把 '40k'/'48000' 等统一成 Hz 整数"""
    s = str(sr).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))


def train_config_path(sr=None) -> Path:
    """按采样率返回训练配置路径；sr 缺省或不受支持时回退到默认 48k。"""
    if sr is not None:
        path = TRAIN_CONFIG_FILES.get(_to_hz(sr))
        if path is not None and path.exists():
            return path
    path = TRAIN_CONFIG_FILES[DEFAULT_TRAIN_SR]
    if not path.exists():
        raise FileNotFoundError(f"找不到训练配置: {path}")
    return path
