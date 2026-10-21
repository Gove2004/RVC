"""模型加载后台线程 — 引擎加载在 QThread 中执行，避免阻塞 UI。"""
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

if TYPE_CHECKING:
    from rvc.streaming.engine import VoiceEngine


class LoadThread(QThread):
    """单次模型加载任务：run() 内调 engine.load_model，结果经 ok/err 信号回主线程。

    支持协作式取消（request_stop）：线程启动前请求则跳过加载；
    已启动后无法中断 CPython 调用，只能等其自然结束。
    """

    ok = Signal(int)
    err = Signal(str)

    def __init__(self, engine: "VoiceEngine", pth: str, hubert: str) -> None:
        super().__init__()
        self.engine = engine
        self.pth = pth
        self.hubert = hubert
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def is_stopping(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            if not self.is_stopping():
                self.ok.emit(self.engine.load_model(
                    self.pth, True, self.hubert))
        except Exception as e:
            self.err.emit(str(e))
