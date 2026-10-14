"""One-shot owner BaseException stops must durably abort even without context."""
from pathlib import Path
import pytest
from anypost import ArchiveSink,ValidationError
from test_storage_prediction import capsule

def fill(sink,c):
    sink.capsule_begin({k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')})
    for case,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(case,weights)
    for key,row in c['stored'].items():sink.capsule_stored(key,row['case_indices'],row['values'])

def final(c):return dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction'])

def test_one_shot_baseexception_after_finalization_aborts_without_context(tmp_path):
    class OwnerStop(BaseException):pass
    stop=OwnerStop('after finalized callback');seen=[];path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    sink=None
    def check():
        if sink.diagnostics['state']=='finalized' and not seen:
            seen.append(True);raise stop
    sink=ArchiveSink(path,cancel_check=check);c=capsule();fill(sink,c)
    temporary=Path(sink._temporary)
    with pytest.raises(OwnerStop) as info:sink.capsule_end(final(c))
    assert info.value is stop and seen==[True]
    with pytest.raises(ValidationError):sink.close()
    assert path.read_bytes()==b'previous' and sink.diagnostics['state']=='aborted'
    assert not temporary.exists() and sink._handle is None

def test_one_shot_keyboard_interrupt_before_close_publish_stays_aborted(tmp_path):
    stop=KeyboardInterrupt('before publish');armed=False;seen=[];path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    def check():
        if armed and not seen:seen.append(True);raise stop
    sink=ArchiveSink(path,cancel_check=check);c=capsule();fill(sink,c);sink.capsule_end(final(c))
    temporary=Path(sink._temporary);armed=True
    with pytest.raises(KeyboardInterrupt) as info:sink.close()
    assert info.value is stop and seen==[True]
    with pytest.raises(ValidationError):sink.close()
    assert path.read_bytes()==b'previous' and sink.diagnostics['state']=='aborted'
    assert not temporary.exists() and sink._handle is None
