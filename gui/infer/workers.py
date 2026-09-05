"""推理 GUI 线程 worker。"""
import logging
import traceback

from PySide6.QtCore import QThread, Signal

from rvc.inference.offline_config import OfflineConfig

logger = logging.getLogger(__name__)


class OfflineWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, cfg: OfflineConfig):
        super().__init__()
        self.cfg = cfg

    def run(self):
        import torch  # 惰性导入，避免 GUI 启动时加载 torch

        try:
            self._do_run()
        except Exception:
            tb = traceback.format_exc()
            logger.error("离线推理失败:\n%s", tb)
            self.error.emit(tb.strip())
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _do_run(self):
        """离线流式推理：模拟播放→转换→写录，复用实时引擎全链路（显存封顶）。"""
        from rvc.audio.realtime_engine import RealtimeEngine

        self.progress.emit(0, 100)

        # OfflineConfig 继承 Params，直接作为运行时参数（音效/音高/破音保护同源）
        engine = RealtimeEngine(self.cfg)
        engine.load_model(self.cfg.model_path, hubert=self.cfg.hubert)
        self.progress.emit(20, 100)

        def _progress(cur, total):
            self.progress.emit(int(cur * 100 / total), 100)

        # 逐块流式处理（内部已完成读音频/重采样/pad/裁剪/RMS/归一化/写录）
        engine.process_file(
            self.cfg.input_path,
            self.cfg.output_path,
            params=self.cfg,
            f0method=self.cfg.f0method,
            protect=self.cfg.protect,
            progress_cb=_progress,
        )
        self.progress.emit(100, 100)
        self.finished.emit(self.cfg.output_path)
