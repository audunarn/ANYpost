"""Bounded, latest-request-wins work with owner-thread publication."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event, RLock
from time import perf_counter_ns
from typing import Any, Callable


@dataclass(frozen=True)
class LoadCompletion:
    request_id: int
    requested_ns: int
    loaded_ns: int
    value: Any = None
    error: BaseException | None = None


class LatestRequestLoader:
    """One I/O worker; only the newest request can publish.

    ``dispatch`` must be thread safe and schedule a callback on the UI owner.
    Cancellation is cooperative: already active filesystem calls complete,
    but their results and callbacks are discarded.
    """

    def __init__(self, dispatch: Callable[[Callable[[], None]], None]):
        self._dispatch = dispatch
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anypost-read")
        self._lock = RLock()
        self._serial = 0
        self._closed = False
        self._future: Future | None = None
        self._completion: LoadCompletion | None = None
        self._cancel = Event()
        self._active: set[Future] = set()
        self.dispatch_error: BaseException | None = None

    @property
    def latest_request_id(self) -> int:
        with self._lock:
            return self._serial

    def request(self, load: Callable[[Event], Any], publish: Callable[[LoadCompletion], None]) -> int:
        requested_ns = perf_counter_ns()
        with self._lock:
            if self._closed:
                raise RuntimeError("result loader is closed")
            self._cancel.set()
            self._completion = None
            if self._future is not None:
                self._future.cancel()
            self._serial += 1
            serial = self._serial
            cancel = self._cancel = Event()
            future = self._executor.submit(load, cancel)
            self._future = future
            self._active.add(future)

        def completed(done: Future) -> None:
            with self._lock:
                self._active.discard(done)
                if self._future is done:
                    self._future = None
                current = not self._closed and self._serial == serial
            if done.cancelled() or cancel.is_set() or not current:
                return
            try:
                completion = LoadCompletion(serial, requested_ns, perf_counter_ns(), value=done.result())
            except BaseException as error:
                completion = LoadCompletion(serial, requested_ns, perf_counter_ns(), error=error)

            with self._lock:
                if self._closed or self._serial != serial or cancel.is_set():
                    return
                self._completion = completion

            def deliver() -> None:
                # Queued callbacks retain only identity, not field arrays.
                # A superseding request or close immediately releases the
                # previous payload even if the UI has not drained its queue.
                with self._lock:
                    eligible = not self._closed and self._serial == serial
                    pending = self._completion if eligible else None
                    if eligible:
                        self._completion = None
                if pending is not None and not cancel.is_set():
                    publish(pending)

            try:
                self._dispatch(deliver)
            except BaseException as error:
                with self._lock:
                    if self._serial == serial:
                        self._completion = None
                    # The callback frame contains the completed Future and
                    # numeric payload. Retain the failure, not that frame.
                    self.dispatch_error = error.with_traceback(None)

        future.add_done_callback(completed)
        return serial

    def close(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                outstanding = tuple(self._active)
            else:
                self._closed = True
                self._serial += 1
                self._cancel.set()
                self._completion = None
                outstanding = tuple(self._active)
        for future in outstanding:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        # shutdown removes queued tasks after Future.cancel(). Those terminal
        # futures can remain CANCELLED (not CANCELLED_AND_NOTIFIED), which
        # concurrent.futures.wait does not recognise as finished on CPython.
        outstanding = tuple(future for future in outstanding if not future.done())
        if outstanding and wait(outstanding, timeout=timeout).not_done:
            raise TimeoutError("result I/O did not retire within the close timeout; keep its reader open")
