"""Byte-accounted LRU; oversized values are returned without retention."""
from collections import OrderedDict
from threading import RLock

class ByteLRU:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.capacity = capacity
        self.resident_bytes = self.peak_bytes = self.hits = self.misses = 0
        self._values = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            item = self._values.pop(key, None)
            if item is None:
                self.misses += 1
                return None
            self._values[key] = item
            self.hits += 1
            return item[0]

    def put(self, key, value, *, key_bytes=0):
        if isinstance(key_bytes, bool) or not isinstance(key_bytes, int) or key_bytes < 0:
            raise ValueError("retained key bytes must be a nonnegative integer")
        size = int(value.nbytes) + key_bytes
        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self.resident_bytes -= previous[1]
            external = sum(getattr(self, "_external", {}).values())
            if size + external > self.capacity:
                return value
            while self.resident_bytes + size + external > self.capacity and self._values:
                _, (_, removed) = self._values.popitem(last=False)
                self.resident_bytes -= removed
            self._values[key] = (value, size)
            self.resident_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes + external)
        return value

    def clear(self):
        with self._lock:
            self._values.clear()
            self.resident_bytes = 0
    def set_external_bytes(self, owner, byte_count):
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("external byte count must be a nonnegative integer")
        with self._lock:
            if not hasattr(self, "_external"):
                self._external = {}
            total = sum(self._external.values()) - self._external.get(owner, 0) + byte_count
            if total > self.capacity:
                from .errors import WorkingSetError
                raise WorkingSetError("retained external buffers exceed cache ceiling")
            if byte_count:
                self._external[owner] = byte_count
            else:
                self._external.pop(owner, None)
            while self.resident_bytes + total > self.capacity and self._values:
                _, (_, removed) = self._values.popitem(last=False)
                self.resident_bytes -= removed
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes + total)
