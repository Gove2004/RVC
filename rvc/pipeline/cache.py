"""推理期共享缓存 — 显式管理 HuBERT / RMVPE / FCPE / Synthesizer 模型复用。

与 `rvc/runtime/lru.py` 的分工：那边是**无业务语义**的通用 LRU 容器，这边是
**带业务语义**的模型槽位管理（保留几个、淘汰谁、淘汰时怎么回收显存）。

LRU 淘汰：synthesizer（几百 MB/个）只进不出会让显存/内存持续增长。
各槽位保留最近使用的 N 个，超限淘汰最久未用的。
槽位数量属于本层（pipeline 组装层）的策略，通用 LRU 基础设施在 runtime。

淘汰即回收：被淘汰的模型若持有 CUDA Graph 缓存，要在这里显式 purge。
CUDAGraph 与其静态输入/输出张量占的显存不会随引用消失立刻归还，而新模型
捕获图时可能复用同一批地址（见 runtime/graph.py 的内存地址竞争说明）——
把回收动作放在「值被遗忘」的那一行，比指望 GC 时机可靠。

失效语义（现状契约）：仅 LRU 容量淘汰，无文件 mtime 校验——
同名 pth 被外部覆盖后仍会返回旧权重，属已知且被接受的现状。
"""
from rvc.runtime.lru import LRUCache
from rvc.runtime.graph import purge_cuda_graphs


def _release_graph_caches(value) -> None:
    """回收被淘汰模型持有的图缓存（bundle 与其内层模型都算 owner）。"""
    if value is None:
        return
    owners = [value]
    inner = getattr(value, "synthesizer", None)
    if inner is not None:
        owners.append(inner)
    purge_cuda_graphs(*owners)


class InferenceCache:
    def __init__(self):
        # 大对象少存：synthesizer 留最近 2 个（切模型时当前+上一个）；
        # hubert 的 key = 设备+精度+variant（base/chinese 各 1 个），留 2 避免来回切换重载；
        # rmvpe/fcpe 的 key 是设备+精度组合（各 1 个），留 1 即可。
        self._hubert = LRUCache(2)
        self._rmvpe = LRUCache(1)
        self._fcpe = LRUCache(1)
        self._synthesizer = LRUCache(2)  # key: (device, is_half, pth_path)
        # 当前活跃的 F0 方法（"rmvpe"/"fcpe"/None），用于避免每块重复调用 release
        self._active_f0_method = None

    def get_hubert(self, key):
        return self._hubert.get(key)

    def set_hubert(self, key, value):
        _release_graph_caches(self._hubert.insert(key, value))

    def get_rmvpe(self, key):
        return self._rmvpe.get(key)

    def set_rmvpe(self, key, value):
        # 提取器自带 release()（清图 + 删模型 + 回收显存），淘汰即调它，
        # 否则槽位容量变化时旧提取器会一直挂着显存
        evicted = self._rmvpe.insert(key, value)
        if evicted is not None:
            evicted.release()

    def get_fcpe(self, key):
        return self._fcpe.get(key)

    def set_fcpe(self, key, value):
        evicted = self._fcpe.insert(key, value)
        if evicted is not None:
            evicted.release()

    def get_synthesizer(self, key):
        """获取缓存的 Synthesizer（LRU 命中即刷新）。

        key 由会话层构造为 (device, is_half, pth_path)——权重是按加载时
        的 dtype/device 落地的，只按路径缓存会在切换后复用错误副本。

        Returns:
            SynthesizerBundle or None
        """
        return self._synthesizer.get(key)

    def set_synthesizer(self, key, value):
        """缓存 Synthesizer。

        Args:
            value: SynthesizerBundle
        """
        _release_graph_caches(self._synthesizer.insert(key, value))

    def release_rmvpe(self) -> None:
        """释放缓存中的 RMVPE 提取器（clear_cuda_graph + del model + gc + empty_cache）。

        切换到 FCPE 时调用，避免两个 F0 模型同时驻留显存。
        """
        for extractor in self._rmvpe.values():
            extractor.release()
        self._rmvpe.clear()

    def release_fcpe(self) -> None:
        """释放缓存中的 FCPE 提取器（clear_cuda_graph + del model + gc + empty_cache）。

        切换到 RMVPE 时调用，避免两个 F0 模型同时驻留显存。
        """
        for extractor in self._fcpe.values():
            extractor.release()
        self._fcpe.clear()

    def switch_f0_method(self, method: str) -> None:
        """切换 F0 方法时释放旧方法的模型（仅在方法真正变化时执行）。

        每块推理都会调用 create_f0_extractor，用 _active_f0_method 标记避免
        每块重复走 release 路径（缓存为空时 release 是 no-op，但仍有锁开销）。
        """
        if self._active_f0_method == method:
            return
        # 释放旧方法
        if self._active_f0_method == "rmvpe":
            self.release_rmvpe()
        elif self._active_f0_method == "fcpe":
            self.release_fcpe()
        self._active_f0_method = method


# 全局单例是有意设计：pipeline 未显式传入 inference_cache 时共享同一份缓存
default_inference_cache = InferenceCache()
