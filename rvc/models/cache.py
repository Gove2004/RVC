"""推理期共享缓存 — 显式管理 HuBERT / RMVPE / FCPE 模型复用。"""
import threading


class InferenceCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._hubert = {}
        self._rmvpe = {}
        self._fcpe = {}

    def get_hubert(self, key):
        with self._lock:
            return self._hubert.get(key)

    def set_hubert(self, key, value):
        with self._lock:
            self._hubert[key] = value

    def get_rmvpe(self, key):
        with self._lock:
            return self._rmvpe.get(key)

    def set_rmvpe(self, key, value):
        with self._lock:
            self._rmvpe[key] = value

    def get_fcpe(self, key):
        with self._lock:
            return self._fcpe.get(key)

    def set_fcpe(self, key, value):
        with self._lock:
            self._fcpe[key] = value


default_inference_cache = InferenceCache()
