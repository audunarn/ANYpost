import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ValidationError
from test_archive_admission import native2
from anypost import ArchiveSink

def stream(path,c):
    with ArchiveSink(path) as sink:
        sink.capsule_begin({k:v for k,v in c.items() if k not in ("cases","coefficients","descriptors","stored")})
        for i,case in enumerate(c["cases"]):sink.capsule_case(case,c["coefficients"][i])
        sink.capsule_end(dict(descriptors=c["descriptors"],associations=c["associations"],responses=c["responses"],derived={},case_count=len(c["cases"]),stored_superseded_cases={}))


def nodal(quantity='displacement',component='UX',key='displacement:UX',stored=False):
    c=native2();old=c['descriptors'][0]['key']
    c['descriptors']=(dict(c['descriptors'][0],key=key,quantity_id=quantity,component=component,location='node',availability='stored' if stored else 'reconstructable'),)
    for name in ('associations','responses','stored'):
        c[name]={key:c[name][old]} if old in c[name] else {}
    if not stored:c['stored']={}
    else:c['responses']={}
    return c


@pytest.mark.parametrize('quantity,key',[('displacement','translation:UX'),('translation','translation:UX'),('displacement','displacement:x')])
@pytest.mark.parametrize('writer',[write_results,stream])
def test_v2_alias_refuses_writer_stream_atomically(tmp_path,quantity,key,writer):
    path=tmp_path/'old.h5';path.write_bytes(b'previous')
    with pytest.raises(ValidationError,match='canonical'):writer(path,nodal(quantity,key=key))
    assert path.read_bytes()==b'previous'


def test_native_metadata_alias_refuses_before_runtime_import(tmp_path):
    path=write_results(tmp_path/'valid.h5',nodal())
    with h5py.File(path,'r+') as h:
        m=json.loads(h['metadata_json'][()].decode());m['descriptors'][0]['quantity_id']='translation'
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(m),dtype=h5py.string_dtype())
    with pytest.raises(ValidationError,match='canonical'):open_results(path)


def test_canonical_reaction_custom_stored_and_legacy_arbitrary_preserved(tmp_path):
    for name,c in [('reaction',nodal('reaction','FX','reaction:FX')),('custom',nodal('translation','UX','translation:UX',stored=True)),('legacy',nodal('translation','UX','translation:UX'))]:
        if name=='legacy':c.update(version=1,runtime='anysolver-results/1');c.pop('combination')
        path=write_results(tmp_path/(name+'.h5'),c)
        with open_results(path) as reader:
            assert reader._runtime is None
            if name=='custom':np.testing.assert_array_equal(reader.read_field('c0','translation','UX'),c['stored']['translation:UX']['values'][0])
