"""推理期共享缓存 — 显式管理 HuBERT / RMVPE / FCPE / Synthesizer 模型复用。

LRU 淘汰：synthesizer（几百 MB/个）只进不出会让显存/内存持续增长。
各槽位保留最近使用的 N 个，超限淘汰最久未用的。
"""
import threading
from collections import OrderedDict


class _LRU:
    """线程安全的 LRU 字典。"""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._d.get(key)
            if value is not None:
                self._d.move_to_end(key)
            return value

    def set(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


class InferenceCache:
    def __init__(self):
        # 大对象少存：synthesizer 留最近 2 个（切模型时当前+上一个）；
        # hubert 的 key = 设备+精度+variant（base/chinese 各 1 个），留 2 避免来回切换重载；
        # rmvpe/fcpe 的 key 是设备+精度组合（各 1 个），留 1 即可。
        self._hubert = _LRU(2)
        self._rmvpe = _LRU(1)
        self._fcpe = _LRU(1)
        self._synthesizer = _LRU(2)  # key: pth_path

    def get_hubert(self, key):
        return self._hubert.get(key)

    def set_hubert(self, key, value):
        self._hubert.set(key, value)

    def get_rmvpe(self, key):
        return self._rmvpe.get(key)

    def set_rmvpe(self, key, value):
        self._rmvpe.set(key, value)

    def get_fcpe(self, key):
        return self._fcpe.get(key)

    def set_fcpe(self, key, value):
        self._fcpe.set(key, value)

    def get_synthesizer(self, key):
        """获取缓存的 Synthesizer（LRU 命中即刷新）。

        Returns:
            SynthesizerBundle or None
        """
        return self._synthesizer.get(key)

    def set_synthesizer(self, key, value):
        """缓存 Synthesizer。

        Args:
            value: SynthesizerBundle
        """
        self._synthesizer.set(key, value)

    def clear_f0_cuda_graph_caches(self):
        """清除所有缓存的 f0 提取器（RMVPE/FCPE）的 CUDA Graph 缓存。

        模型重加载时调用：synthesizer/hubert 的 CUDA Graph 在 load_model_session 中清除，
        但 f0 提取器通过 inference_cache 独立缓存，旧 CUDA Graph 残留可能导致
        快速 stop/start 后 f0 提取异常（声音沙哑）。
        """
        from rvc.tools.cuda_graph import clear_cuda_graph_cache
        for lru in (self._rmvpe, self._fcpe):
            with lru._lock:
                for extractor in lru._d.values():
                    # RMVPEExtractor.model 是 RMVPE 对象，CUDA Graph 在 model.mel_extractor 和 model.model 上
                    if hasattr(extractor, 'model'):
                        if hasattr(extractor.model, 'mel_extractor'):
                            clear_cuda_graph_cache(extractor.model.mel_extractor)
                        clear_cuda_graph_cache(extractor.model)


default_inference_cache = InferenceCache()
