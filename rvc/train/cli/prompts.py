"""交互式提问与通用格式化工具。

提示文案格式契约（`_hint`）：`问题（说明；可选 a / b；默认 x）: ` ——
说明/可选项/默认值收进**同一对括号**（`；`分隔，顺序固定：说明在先）。
提问文本本身不许带括号；单位写在 note 开头（如「秒。…」）。
"""
from rvc.train.cli import Cancelled, EnvFatal

_YES = ("y", "yes", "是", "1")
_NO = ("n", "no", "否", "0")


def _clean(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def _hint(default="", options=None, note="") -> str:
    """问题旁的括号提示：补充说明 + 可选项 + 默认值（同括号内，分号分隔）。"""
    parts = []
    if note:
        parts.append(note.strip().rstrip("；;"))
    if options:
        parts.append("可选 " + " / ".join(str(o) for o in options))
    if default != "":
        parts.append(f"默认 {default}")
    else:
        parts.append("必填")
    return f"（{'；'.join(parts)}）"


def ask(prompt: str, default="", cast=None, check=None,
        options=None, note="") -> object:
    """提问。default 为空串时视为必填；EOF（管道结束）自动取默认值。

    options 仅用于在问题旁提示可选项，校验仍由 check 负责（两者不要各写一份）。
    """
    while True:
        try:
            raw = _clean(input(f"{prompt}{_hint(default, options, note)}: "))
        except EOFError:
            print(flush=True)
            if default != "":
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


def ask_yes(prompt: str, default: bool = True, note: str = "") -> bool:
    while True:
        hint = _hint("y" if default else "n", ("y", "n"), note)
        try:
            raw = _clean(input(f"{prompt}{hint}: ")).lower()
        except EOFError:
            print(flush=True)
            return default
        except KeyboardInterrupt:
            print(flush=True)
            raise Cancelled() from None
        if raw == "":
            return default
        if raw in _YES:
            return True
        if raw in _NO:
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
