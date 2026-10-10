"""通用线程安全 LRU 缓存。

为什么在 runtime 层：模型实例复用（pipeline 槽位策略）与图缓存都要 LRU，
它是无业务语义的基础设施。槽位保留数（hubert=2 / rmvpe=1 / synthesizer=2）
属于 pipeline 组装层的策略，不在这里。
"""
import threading
from collections import OrderedDict


class LRUCache:
    """线程安全的 LRU 字典。"""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        """按键取值并刷新 LRU 位置；未命中返回 default。

        用**成员判定**而非 `self._d.get(key) is None`：后者会把「值本身是 None」
        与「未命中」混为一谈，并且不会为 None 值刷新 LRU 位置。
        """
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return default

    def set(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def values(self):
        """线程安全地返回所有值的快照列表。"""
        with self._lock:
            return list(self._d.values())

    def clear(self):
        """清空所有条目（不释放值本身，调用方需先 release）。"""
        with self._lock:
            self._d.clear()
