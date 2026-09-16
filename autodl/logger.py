"""训练日志：控制台 + 文件双写，支持实验名确定后 reattach。"""
import time
from pathlib import Path

from autodl import BAR


class TrainLogger:
    """控制台 + 日志文件。

    启动即写到引导日志（输出根/logs/autodl_train.log），实验名确定后 reattach 到
    <实验目录>/autodl_train.log，并把引导阶段的内容原样搬到新文件头部，最终只留一份。
    """

    def __init__(self):
        self.log_file = None
        self._fp = None
        self._buffer = []
        self._t0 = time.time()

    def attach(self, log_file: Path):
        self.log_file = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_file.open("a", encoding="utf-8")
        for level, msg in self._buffer:
            self._write(level, msg)
        self._buffer.clear()

    def reattach(self, log_file: Path):
        """把日志搬到新位置，旧内容搬到新文件头部。"""
        old_file, old_fp = self.log_file, self._fp
        carried = []
        try:
            if old_fp:
                old_fp.close()
            if old_file and old_file.exists() and old_file != log_file:
                carried = old_file.read_text(encoding="utf-8").splitlines()
                old_file.unlink()
        except Exception:
            pass
        self._buffer.clear()
        self.attach(log_file)
        if carried:
            self._fp.write("—— 以下为实验名确定前的体检记录 ——\n")
            for line in carried:
                self._fp.write(line + "\n")
            self._fp.write("——————————————————\n")
            self._fp.flush()

    @staticmethod
    def _stamp():
        return time.strftime("%m-%d %H:%M:%S")

    def _write(self, level, msg):
        self._fp.write(f"[{self._stamp()}] [{level:<5}] {msg}\n")
        self._fp.flush()

    def log(self, msg: str, level: str = "INFO"):
        line = f"[{self._stamp()}] [{level:<5}] {msg}" if self._fp else msg
        print(line, flush=True)
        if self._fp:
            self._write(level, msg)
        else:
            self._buffer.append((level, msg))

    def file_only(self, msg: str, level: str = "INFO"):
        """只写日志、不上屏（Trainer 自带的 epoch loss 行与下面的 EPOCH 行重复）。"""
        if self._fp:
            self._write(level, msg)
        else:
            self._buffer.append((level, msg))

    def plain(self, msg: str = ""):
        """只上屏、不进日志的分隔/提示行。"""
        print(msg, flush=True)

    def section(self, title: str):
        self.plain(BAR)
        self.log(f"【{title}】")
        self.plain(BAR)

    def elapsed(self):
        return time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - self._t0))

    def close(self):
        try:
            if self._fp:
                self._fp.close()
        except Exception:
            pass
