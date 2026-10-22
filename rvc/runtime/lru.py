"""通用线程安全 LRU 缓存。

为什么在 runtime 层：模型实例复用（pipeline 槽位策略）与图缓存都要 LRU，
它是无业务语义的基础设施。槽位保留数（hubert=2 / rmvpe=1 / synthesizer=2）
属于 pipeline 组装层的策略，不在这里。
"""
import threading
from collections import OrderedDict


class LRUCache:
    """线程安全的 LRU 字典。"""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: object, default: object = None) -> object:
        """按键取值并刷新 LRU 位置；未命中返回 default。

        用**成员判定**而非 `self._d.get(key) is None`：后者会把「值本身是 None」
        与「未命中」混为一谈，并且不会为 None 值刷新 LRU 位置。
        """
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return default

    def insert(self, key: object, value: object) -> object:
        """写入并刷新 LRU 位置；因容量淘汰时返回被淘汰的值（未淘汰返回 None）。

        返回淘汰值是给调用方的回收钩子：值可能持有 GPU 资源（如 CUDA Graph
        缓存），必须在被遗忘的那一刻显式回收，不能指望 GC 时机。
        """
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            if len(self._d) > self.maxsize:
                _, evicted = self._d.popitem(last=False)
                return evicted
            return None

    def values(self) -> list:
        """线程安全地返回所有值的快照列表。"""
        with self._lock:
            return list(self._d.values())

    def clear(self) -> None:
        """清空所有条目（不释放值本身，调用方需先 release）。"""
        with self._lock:
            self._d.clear()
