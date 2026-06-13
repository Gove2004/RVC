"""推理期共享缓存 — 显式管理 HuBERT / RMVPE / FCPE / Synthesizer / Index 模型复用。"""
import threading


class InferenceCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._hubert = {}
        self._rmvpe = {}
        self._fcpe = {}
        self._synthesizer = {}  # key: pth_path
        self._index = {}        # key: index_path

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

    def get_synthesizer(self, key):
        """获取缓存的 Synthesizer。

        Args:
            key: 模型路径

        Returns:
            dict or None: {"net_g": model, "tgt_sr": int, "if_f0": int, "version": str}
        """
        with self._lock:
            return self._synthesizer.get(key)

    def set_synthesizer(self, key, value):
        """缓存 Synthesizer。

        Args:
            key: 模型路径
            value: dict: {"net_g": model, "tgt_sr": int, "if_f0": int, "version": str}
        """
        with self._lock:
            self._synthesizer[key] = value

    def get_index(self, key):
        """获取缓存的 FAISS Index。

        Args:
            key: index 文件路径

        Returns:
            dict or None: {"index": faiss.Index, "big_npy": np.ndarray}
        """
        with self._lock:
            return self._index.get(key)

    def set_index(self, key, value):
        """缓存 FAISS Index。

        Args:
            key: index 文件路径
            value: dict: {"index": faiss.Index, "big_npy": np.ndarray}
        """
        with self._lock:
            self._index[key] = value


default_inference_cache = InferenceCache()
