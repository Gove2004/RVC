"""推理期共享缓存 — 显式管理 HuBERT / RMVPE / FCPE / Synthesizer 模型复用。

LRU 淘汰：synthesizer（几百 MB/个）只进不出会让显存/内存持续增长。
各槽位保留最近使用的 N 个，超限淘汰最久未用的。
槽位数量属于本层（pipeline 组装层）的策略，通用 LRU 基础设施在 runtime。

失效语义（现状契约，测试锁定）：仅 LRU 容量淘汰，无文件 mtime 校验——
同名 pth 被外部覆盖后仍会返回旧权重，属已知且被接受的现状。
"""
from rvc.runtime.caches import LRUCache
from rvc.runtime.graph import purge_cuda_graphs


class InferenceCache:
    def __init__(self):
        # 大对象少存：synthesizer 留最近 2 个（切模型时当前+上一个）；
        # hubert 的 key = 设备+精度+variant（base/chinese 各 1 个），留 2 避免来回切换重载；
        # rmvpe/fcpe 的 key 是设备+精度组合（各 1 个），留 1 即可。
        self._hubert = LRUCache(2)
        self._rmvpe = LRUCache(1)
        self._fcpe = LRUCache(1)
        self._synthesizer = LRUCache(2)  # key: pth_path

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

        模型重加载时调用：synthesizer/hubert 的 CUDA Graph 在 ModelSessions 中清除，
        但 f0 提取器通过 inference_cache 独立缓存，旧 CUDA Graph 残留可能导致
        快速 stop/start 后 f0 提取异常（声音沙哑）。
        """
        # 调用各提取器的抽象基类方法 clear_cuda_graph()，
        # 不直接访问 model.mel_extractor / model.model 等内部属性
        for extractor in self._rmvpe.values():
            extractor.clear_cuda_graph()
        for extractor in self._fcpe.values():
            extractor.clear_cuda_graph()

    def clear_synthesizer_cuda_graphs(self) -> None:
        """清除所有缓存的 synthesizer 的 CUDA Graph（外部经此操作，不触碰私有槽位）。"""
        purge_cuda_graphs(
            *(bundle.synthesizer for bundle in self.synthesizer_bundles())
        )

    def synthesizer_bundles(self) -> tuple:
        """当前缓存的全部 synthesizer bundle 快照（LRU 最新→最旧）。"""
        return tuple(self._synthesizer.values())


default_inference_cache = InferenceCache()
