"""交互式提问与通用格式化工具。"""
from autodl import Cancelled, EnvFatal


def _clean(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def ask(prompt: str, default="", cast=None, check=None, allow_blank=False):
    """提问。default 为空串 = 必填；EOF（管道结束）自动取默认值。"""
    while True:
        hint = f"（默认 {default}）" if default != "" else ""
        try:
            raw = _clean(input(f"{prompt}{hint}: "))
        except EOFError:
            print(flush=True)
            if allow_blank or default != "":
                raw = ""
            else:
                raise EnvFatal(
                    f"必填项未填写，且输入已结束: {prompt}\n"
                    "  提示：非交互运行时把答案按行喂给 stdin，或把数据集路径作为第一个参数传入"
                ) from None
        except KeyboardInterrupt:
            print(flush=True)
            raise Cancelled() from None
        if raw == "":
            if allow_blank:
                return ""
            if default != "":
                raw = str(default)
            else:
                print("  × 该项必填，请重新输入", flush=True)
                continue
        if cast is not None:
            try:
                val = cast(raw)
            except Exception:
                print(f"  × 格式不对：{raw}", flush=True)
                continue
        else:
            val = raw
        if check is not None:
            ok, msg = check(val)
            if not ok:
                print(f"  × {msg}", flush=True)
                continue
        return val


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = _clean(input(f"{prompt} [{hint}]: ")).lower()
        except EOFError:
            print(flush=True)
            return default
        except KeyboardInterrupt:
            print(flush=True)
            raise Cancelled() from None
        if raw == "":
            return default
        if raw in ("y", "yes", "是", "1"):
            return True
        if raw in ("n", "no", "否", "0"):
            return False
        print("  × 请输入 y 或 n", flush=True)


def _as_int(raw):
    return int(str(raw).strip())


def _as_float(raw):
    return float(str(raw).strip())


def _as_sr(raw):
    """接受 48k / 48 / 48000 三种写法，统一返回 '48k'。"""
    text = str(raw).strip().lower().replace("hz", "")
    if text.endswith("k"):
        return f"{_as_int(text[:-1])}k"
    value = _as_int(text)
    return f"{value // 1000}k" if value >= 1000 else f"{value}k"


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024


def _human_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
