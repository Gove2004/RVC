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

    def values(self):
        """线程安全地返回所有值的快照列表。"""
        with self._lock:
            return list(self._d.values())
