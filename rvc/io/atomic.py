"""文件原子写入 — 先写独占的 .tmp 再 os.replace，任何时刻磁盘上的文件都完整。

中断不留截断文件：按「文件存在即有效」消费的产物（checkpoint/配置/存档）
若被中途强杀（Ctrl+C 强退/断电）留下半个文件，会被静默复用并损坏。
replace 是同文件系统原子操作，写完后目标要么是旧内容要么是新内容。

**并发安全（同一路径可被多进程同时写）**：tmp 名带 pid 与自增序号，每个写者
独占自己的 tmp，因此互不截断。这是必需的而非可选——训练侧 DataLoader 的多个
worker 是独立进程，会并发首次填充同一批样本缓存（dataset._cached_wav/_cached_spec
惰性写盘、无预热 pass），固定 tmp 名会让先完成者把 tmp 移走、后到者在
os.replace 处抛 FileNotFoundError（实测约 20% 触发率），且只重试 PermissionError
的退避逻辑兜不住它，表现为「新实验第一个 epoch 偶发崩溃、重跑又好」。
"""
import itertools
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# 同一进程内的写者用自增序号区分（线程并发写同一路径也不会撞 tmp 名）。
_tmp_counter = itertools.count()

# os.replace 的重试次数与退避基数（秒）。Windows 上目标文件被短暂占用时
# replace 抛 PermissionError，退避总时长约 0.05+0.10+0.15+0.20 = 0.5s。
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF = 0.05


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """os.replace，Windows 上对 PermissionError 做有限重试。

    两个平台语义不同：POSIX 的 rename 只改目录项，目标被打开也能成功；Windows
    的 MoveFileEx 在目标被任何进程以不共享删除的方式打开时会失败。训练产物
    （.pth/.pt）正是杀毒软件、Everything 索引器、资源管理器预览最常扫描的
    文件——长训练里偶发一次 save 失败就会丢掉整轮权重。

    只重试 PermissionError：ENOSPC/EROFS 之类的真实错误必须立刻抛出。
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF * (attempt + 1))


def _unique_tmp(path: Path) -> Path:
    """本写者独占的临时路径（pid + 进程内自增序号）。

    不能用固定的 `path + ".tmp"`：多进程写同一目标时会互相截断同一个 tmp，
    先完成者的 os.replace 把 tmp 移走，后到者随即 FileNotFoundError。
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{next(_tmp_counter)}.tmp")


def atomic_write(path: str | Path, writer: Callable[[Any], None]) -> None:
    """原子写。writer 接收已打开的二进制文件对象，负责序列化。

    失败（含 writer 抛错）时清理本写者自己的 tmp，不留垃圾；目标是共享的，
    但 tmp 是独占的，所以清理不会影响其他写者。
    """
    path = Path(path)
    tmp = _unique_tmp(path)
    try:
        with open(tmp, "wb") as f:
            writer(f)
        _replace_with_retry(tmp, path)
    except BaseException:
        # 只在 tmp 仍存在时删（replace 成功后它已不存在；竞态下由 OSError 兜住）
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """原子写文本。"""
    atomic_write(path, lambda f: f.write(text.encode(encoding)))


def load_json_or_default(path: str | Path, default: dict, *,
                         logger: logging.Logger) -> dict:
    """读 JSON dict；文件缺失或损坏时返回 default。

    损坏不静默吞：先把原文件备份成 .bak（用户数据可手工找回），再回退
    default——直接返回空 dict 会让用户的模型列表/偏好静默丢失且无法恢复。
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            os.replace(path, backup)
            logger.warning("JSON 文件损坏，已备份到 %s：%s", backup, e)
        except Exception:
            logger.warning("JSON 文件损坏且备份失败：%s", e, exc_info=True)
        return default
    return data if isinstance(data, dict) else default
