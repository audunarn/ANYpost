from queue import Queue
from threading import Event
import time
import pytest
from anypost.async_loader import LatestRequestLoader


def until(predicate, timeout=2):
    deadline = time.monotonic()+timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.002)


def test_newest_request_discards_active_and_queued_predecessors():
    dispatched, published = Queue(), []
    started, release = Event(), Event()
    loader = LatestRequestLoader(dispatched.put)
    def slow(cancel):
        started.set()
        assert release.wait(2)
        return "obsolete"
    loader.request(slow, published.append)
    assert started.wait(2)
    loader.request(lambda cancel: "also obsolete", published.append)
    newest = loader.request(lambda cancel: "latest", published.append)
    release.set()
    until(lambda: not dispatched.empty())
    while not dispatched.empty():
        dispatched.get_nowait()()
    assert [(v.request_id, v.value) for v in published] == [(newest, "latest")]
    loader.close()


def test_dispatched_completion_rechecks_revision_and_close():
    dispatched, published = Queue(), []
    loader = LatestRequestLoader(dispatched.put)
    loader.request(lambda cancel: "first", published.append)
    until(lambda: not dispatched.empty())
    first = dispatched.get_nowait()
    loader.request(lambda cancel: "second", published.append)
    first()
    assert not published
    until(lambda: not dispatched.empty())
    second = dispatched.get_nowait()
    loader.close()
    second()
    assert not published
    with pytest.raises(RuntimeError):
        loader.request(lambda cancel: None, published.append)


def test_failure_is_delivered_without_becoming_success():
    dispatched, published = Queue(), []
    loader = LatestRequestLoader(dispatched.put)
    def fail(cancel):
        raise LookupError("missing component")
    loader.request(fail, published.append)
    until(lambda: not dispatched.empty())
    dispatched.get_nowait()()
    assert isinstance(published[0].error, LookupError)
    assert published[0].value is None
    loader.close()


def test_published_field_payload_is_released():
    import gc
    import weakref
    import numpy as np
    dispatched, refs = Queue(), []
    loader = LatestRequestLoader(dispatched.put)
    def load(cancel):
        field = np.zeros(1024, dtype=np.float64)
        refs.append(weakref.ref(field))
        return field
    loader.request(load, lambda completion: None)
    callback = dispatched.get(timeout=2)
    callback()
    del callback
    gc.collect()
    assert refs[0]() is None
    loader.close()


def test_queued_superseded_and_closed_fields_release_before_ui_drain():
    import gc
    import weakref
    import numpy as np
    dispatched, refs, published = Queue(), [], []
    loader = LatestRequestLoader(dispatched.put)
    def load(cancel):
        field = np.zeros(1024, dtype=np.float64)
        refs.append(weakref.ref(field))
        return field
    loader.request(load, published.append)
    until(lambda: dispatched.qsize() == 1)
    loader.request(load, published.append)
    until(lambda: dispatched.qsize() == 2)
    gc.collect()
    assert refs[0]() is None
    assert refs[1]() is not None
    loader.close()
    gc.collect()
    assert refs[1]() is None
    while not dispatched.empty():
        dispatched.get_nowait()()
    assert not published


def test_dispatch_failure_releases_field_and_retains_error():
    import gc
    import weakref
    import numpy as np
    refs = []
    def dispatch(callback):
        raise RuntimeError("owner queue is closed")
    loader = LatestRequestLoader(dispatch)
    def load(cancel):
        field = np.zeros(1024, dtype=np.float64)
        refs.append(weakref.ref(field))
        return field
    loader.request(load, lambda completion: None)
    until(lambda: loader.dispatch_error is not None)
    gc.collect()
    assert refs[0]() is None
    assert str(loader.dispatch_error) == "owner queue is closed"
    loader.close()



def test_close_retires_cancelled_pending_future_without_wait_notification_timeout():
    from threading import Event,Timer
    entered,release=Event(),Event();ran=[];published=[]
    loader=LatestRequestLoader(lambda callback:callback())
    def active(cancel):entered.set();release.wait(2);return 'retired'
    loader.request(active,published.append);assert entered.wait(1)
    loader.request(lambda cancel:ran.append('pending'),published.append)
    pending=loader._future
    timer=Timer(.02,release.set);timer.start()
    try:loader.close(timeout=1)
    finally:release.set();timer.join();loader.close(timeout=1)
    assert pending.cancelled() and pending.done() and ran==[] and published==[]


def test_close_still_refuses_unretired_active_work_and_can_complete_later():
    from threading import Event
    entered,release=Event(),Event();loader=LatestRequestLoader(lambda callback:callback())
    def active(cancel):entered.set();release.wait(2)
    loader.request(active,lambda completion:None);assert entered.wait(1)
    try:
        import pytest
        with pytest.raises(TimeoutError):loader.close(timeout=.01)
    finally:release.set();loader.close(timeout=1)
