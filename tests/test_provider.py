"""Independent provider-boundary and native-sample regressions."""
from threading import Event
import numpy as np
import pytest
from anypost import (open_memory_results,reduce_samples,ValidationError,IdentityError,
    FailedCaseError,UnavailableFieldError,WorkingSetError,ClosedReaderError,RequestCancelled,UnknownCaseError)

class Provider:
    def __init__(self):
        self.calls=[];self.closed=False
        self.ids=np.array([41,9,700],dtype=np.int64)
        self.rows={'a':np.array([[-2.,2.,99.],[3.,np.nan,88.],[0.,77.,66.]]),
                   'b':np.array([[4.,-1.,55.],[5.,np.nan,44.],[0.,22.,33.]]),
                   'failed':np.full((3,3),1e9)}
    def metadata(self):
        return {'model_identity':'original-model','mesh_identity':'original-mesh','units':{'stress':'Pa'},
            'cases':[{'id':'a','status':'completed','input_index':8,'certified':False},
                     {'case_id':'b','status':'converged','input_index':2},
                     {'id':'failed','status':'failed','input_index':1,'diagnostics':{'why':'retained source refusal'}}],
            'descriptors':[{'quantity_id':'stress','components':('xx',),'location':'element','unit':'Pa',
                'basis':'local','ids':self.ids,'sample_counts':np.array([2,2,0],dtype=np.int64),
                'value_shape':(3,),'sampling':'native_source','provider_work_bytes':0}]}
    def read_field(self,cid,q,c,selection=None,cancel_event=None):
        self.calls.append((cid,q,c,selection));return self.rows[cid][selection]
    def close(self):self.closed=True

def test_discovery_no_load_and_identity_metadata_alias_isolation():
    p=Provider()
    with open_memory_results(p,expected_model_identity='original-model') as r:
        assert not p.calls
        assert r.cases()[0]['case_id']=='a' and r.case_metadata('failed')['diagnostics']['why']=='retained source refusal'
        m=r.metadata();assert m['descriptors'][0]['components']==('xx',)
        with pytest.raises(TypeError):m['units']['stress']='bad'
        p.ids[0]=999
        assert r.components()[0]['ids'][0]==41
        m['descriptors'][0]['ids'].shape=(3,1)
        assert r.components()[0]['ids'].tolist()==[41,9,700]
    assert not p.closed
    with pytest.raises(ClosedReaderError):r.read_field('a','stress','xx')
    with pytest.raises(IdentityError):open_memory_results(p,expected_model_identity='another-model')

def test_native_selection_cache_identity_precision_and_readonly_snapshots():
    p=Provider()
    with open_memory_results(p) as r:
        result=r.read_field('a','stress','xx',[2,0,0]);np.testing.assert_array_equal(result,p.rows['a'][[2,0,0]])
        assert result.dtype==np.float64
        with pytest.raises(ValueError):result.setflags(write=True)
        before=result.tobytes();result.shape=(9,)
        cached=r.read_field('a','stress','xx',[2,0,0]);assert cached.shape==(3,3) and cached.tobytes()==before
        assert len(p.calls)==1
        p.rows['a'][:]=999
        assert r.read_field('a','stress','xx',[2,0,0]).tobytes()==before
        r.clear_cache();assert r.read_field('a','stress','xx',[2,0,0])[0,0]==999
        np.testing.assert_array_equal(r.associations('stress','xx',[2,0,0]),[700,41,41])

def test_negative_slice_and_range_versus_explicit_index_cache_collision():
    p=Provider()
    with open_memory_results(p) as r:
        np.testing.assert_array_equal(r.read_field('a','stress','xx',slice(None,None,-1)),p.rows['a'][::-1])
        sliced=r.read_field('a','stress','xx',slice(0,3,2))
        explicit=r.read_field('a','stress','xx',[0,2,2])
        assert len(sliced)==2 and len(explicit)==3 and len(p.calls)==3
        with pytest.raises(ValidationError):r.read_field('a','stress','xx',[3])

def test_failed_missing_unknown_cases_are_distinct():
    p=Provider()
    with open_memory_results(p) as r:
        with pytest.raises(FailedCaseError):r.read_field('failed','stress','xx')
        with pytest.raises(UnknownCaseError):r.read_field('never','stress','xx')
        with pytest.raises(UnavailableFieldError):r.read_field('a','stress','missing')
        assert not p.calls

@pytest.mark.parametrize('policy,expected',[('propagate',np.nan),('omit',3.)])
def test_ragged_counts_nan_policy_signed_first_max_abs(policy,expected):
    values=np.array([[-2.,2.,999.],[3.,np.nan,777.],[1.,2.,3.]])
    result=reduce_samples(values,'max_abs',sample_counts=np.array([2,2,0]),nan_policy=policy)
    assert result[0]==-2. and np.isnan(result[2])
    if np.isnan(expected):assert np.isnan(result[1])
    else:assert result[1]==expected
    assert reduce_samples(values,'mean',sample_counts=np.array([2,1,0]),empty_policy='zero').tolist()==[0.,3.,0.]
    with pytest.raises(ValidationError):reduce_samples(values,'mean',sample_counts=np.array([2,2,0]),nan_policy='raise')
    with pytest.raises(UnavailableFieldError):reduce_samples(values,'mean',sample_counts=np.array([2,1,0]),empty_policy='raise')
    with pytest.raises(ValidationError):reduce_samples(values,'mean',sample_counts=np.array([4,1,1]))

def test_multiaxis_explicit_native_c_order_and_reduction_selectors():
    tensor=np.array([[[-4.,4.],[1.,2.]],[[2.,3.],[4.,5.]]])
    assert reduce_samples(tensor,'max_abs').tolist()==[-4.,5.]
    p=Provider()
    with open_memory_results(p) as r:
        result=r.read_field('a','stress','xx',[2,0,0],reduction='max_abs',empty_policy='zero')
        assert result.tolist()==[0.,-2.,-2.]

def test_envelope_source_cases_near_ties_failures_and_authoritative_order():
    p=Provider();p.rows['a']=np.array([[1.,-1.,0.],[2.,0.,0.],[3.,0.,0.]])
    p.rows['b']=np.array([[np.nextafter(1.,np.inf),0.,0.],[2.,0.,0.],[3.,0.,0.]])
    with open_memory_results(p) as r:
        envelope=r.envelope('stress','xx',[0,1],case_ids=['b','failed','a'],reduction='max',tie_policy='exact_earliest_input_index')
        assert envelope.failed_case_ids==('failed',) and envelope.requested_case_ids==('b','failed','a')
        assert envelope.max_case_ids.tolist()==['b','b'] and envelope.min_case_ids.tolist()==['a','b']
        first=r.read_envelope('stress','xx',[1],case_ids=['a','b'],mode='max',reduction='max',tie_policy='first')
        last=r.read_envelope('stress','xx',[1],case_ids=['a','b'],mode='max',reduction='max',tie_policy='last')
        assert first.case_ids.tolist()==['a'] and last.case_ids.tolist()==['b']
        with pytest.raises(FailedCaseError):r.envelope('stress','xx',strict_failed=True)
        signed=r.read_envelope('stress','xx',[0],case_ids=['a','b'],mode='max_abs',reduction='max_abs')
        assert signed.case_ids.tolist()==['b']

def test_first_case_nan_envelope_has_no_false_finite_governing_id():
    p=Provider();p.rows['a'][0,0]=np.nan
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[0],case_ids=['a','b'],mode='max',reduction='mean')
        assert np.isnan(result.values[0]) and result.case_ids[0]==''

def test_missing_successful_cases_refused_with_coverage_and_empty_not_absent():
    class Missing(Provider):
        def read_field(self,cid,*args,**kwargs):raise KeyError(cid)
    with open_memory_results(Missing()) as r:
        with pytest.raises(UnavailableFieldError) as caught:r.envelope('stress','xx',[0],case_ids=['a','b'])
        assert caught.value.unavailable_case_ids==('a','b')
    p=Provider()
    with open_memory_results(p) as r:
        assert r.read_field('a','stress','xx',[2],reduction='mean',empty_policy='zero')[0]==0
        with pytest.raises(UnavailableFieldError):r.read_field('a','stress','absent',[2],reduction='mean',empty_policy='zero')

def test_eviction_external_reservations_and_work_refusal_before_source():
    p=Provider()
    with open_memory_results(p,cache_bytes=80,working_bytes=1<<20) as r:
        for cid in ('a','b','a'):r.read_field(cid,'stress','xx',[0])
        assert len(p.calls)==3 and r.diagnostics['cache_resident_bytes']<=80
        r.set_external_cache_bytes('display',16);r.clear_cache()
        assert r.diagnostics['external_cache_bytes']==64
        with pytest.raises(WorkingSetError):r.set_external_cache_bytes('display',1000)
    class Cost(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0]['provider_work_bytes']=1<<20;return m
    p=Cost()
    with open_memory_results(p,working_bytes=65536) as r:
        with pytest.raises(WorkingSetError):r.read_field('a','stress','xx',[0])
        assert not p.calls
    with pytest.raises(WorkingSetError):reduce_samples(np.ones((100,3)),working_bytes=1)

def test_cancellation_before_after_provider_and_close_does_not_close_owner():
    token=Event();token.set();p=Provider()
    with open_memory_results(p) as r:
        with pytest.raises(RequestCancelled):r.read_field('a','stress','xx',cancel_event=token)
        assert not p.calls
        token.clear()
        original=p.read_field
        def cancelled(*args,**kwargs):token.set();return original(*args,**kwargs)
        p.read_field=cancelled
        with pytest.raises(RequestCancelled):r.read_field('a','stress','xx',cancel_event=token)
        assert r.diagnostics['cache_resident_bytes']==48
    assert not p.closed

def test_dynamic_case_layout_and_optional_provider_calls():
    class Dynamic(Provider):
        provider_work_bytes=4096
        def field_descriptor(self,cid,q,c):return {'sample_counts':np.array([1,1,0]),'value_shape':(3,),'provider_work_bytes':0}
        def read_history(self,cid):return {'values':self.rows[cid],'time':np.arange(3,dtype=float)}
        def query(self,name):return {'table':np.array([1.,2.]),'name':name}
        def geometry(self,selection=None):return {'positions':np.zeros((3,3)), 'triangles':np.empty((0,3),dtype=np.int64)}
    p=Dynamic()
    with open_memory_results(p) as r:
        assert not p.calls
        assert r.read_field('a','stress','xx',[0],reduction='mean')[0]==-2.
        assert r.query('loads')['name']=='loads'
        assert r.read_history('a')['values'].shape==(3,3)
        assert r.geometry()['positions'].shape==(3,3)

def test_malformed_provider_declared_array_and_metadata_refusal():
    p=Provider();p.rows['a']=p.rows['a'].astype(np.float32)
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.read_field('a','stress','xx')
    p=Provider();p.rows['a'][0,0]=np.inf
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.read_field('a','stress','xx')
    class Bad(Provider):
        def metadata(self):m=super().metadata();m['cases'].append(m['cases'][0]);return m
    with pytest.raises(ValidationError):open_memory_results(Bad())


def test_explicit_translation_magnitude_and_stress_triplet_refusal():
    vectors=np.array([[3.,4.,0.],[-2.,0.,0.]])
    np.testing.assert_array_equal(reduce_samples(vectors,'magnitude',vector_components=('ux','uy','uz')),np.linalg.norm(vectors,axis=1))
    with pytest.raises(ValidationError):reduce_samples(vectors,'magnitude')
    with pytest.raises(ValidationError):reduce_samples(np.ones((2,3,1)),'magnitude',vector_components=('ux','uy','uz'))


def test_dynamic_descriptor_work_is_admitted_before_provider_callback():
    class Cost(Provider):
        descriptor_calls=0
        def metadata(self):m=super().metadata();m['descriptors'][0]['provider_work_bytes']=1<<20;return m
        def field_descriptor(self,*args):self.descriptor_calls+=1;raise AssertionError('must not recover')
    p=Cost()
    with open_memory_results(p,working_bytes=65536) as r:
        with pytest.raises(WorkingSetError):r.read_field('a','stress','xx',[0])
    assert p.descriptor_calls==0 and not p.calls


def test_explicit_entity_presence_is_not_source_nan_or_present_empty_zero():
    class Present(Provider):
        def field_descriptor(self,cid,q,c):return {'present':np.array([cid=='b',True,cid=='a'])}
    p=Present()
    with open_memory_results(p) as r:
        assert np.isnan(r.read_field('a','stress','xx',[0],reduction='mean',empty_policy='zero')[0])
        result=r.read_envelope('stress','xx',[0,2],case_ids=['a','b'],reduction='mean',empty_policy='zero')
        assert result.values.tolist()==[1.5,0.] and result.case_ids.tolist()==['b','a']
        # A present actual NaN is not skipped in favour of a later finite row.
        result=r.read_envelope('stress','xx',[1],case_ids=['a','b'],reduction='mean')
        assert np.isnan(result.values[0]) and result.case_ids[0]==''


def test_resize_cache_evicts_only_ordinary_entries_and_rejects_atomically():
    p=Provider()
    with open_memory_results(p,cache_bytes=200) as r:
        r.read_field('a','stress','xx',[0]);r.read_field('b','stress','xx',[0])
        r.resize_cache(80)
        assert r.diagnostics['cache_resident_bytes']<=80
        r.read_field('a','stress','xx',[0]);assert len(p.calls)==3
        r.set_external_cache_bytes('display',16)
        before=r.diagnostics.copy()
        with pytest.raises(WorkingSetError):r.resize_cache(10)
        assert r.diagnostics['cache_bytes']==before['cache_bytes']
        assert r.diagnostics['cache_resident_bytes']==before['cache_resident_bytes']
        with pytest.raises(ValueError):r.resize_cache(True)


def test_generic_rendering_vector_norm_and_absolute_native_samples():
    from anypost import vector_magnitude,absolute_sample_max
    values=np.array([[3.,4.],[-5.,12.]])
    np.testing.assert_array_equal(vector_magnitude(values,components=('stored_x','stored_y')),np.linalg.norm(values,axis=-1))
    with pytest.raises(ValidationError):vector_magnitude(values,components=('only_one',))
    samples=np.array([[np.nan,-3.,2.],[np.nan,np.nan,np.nan]])
    out=absolute_sample_max(samples)
    assert out[0]==3. and np.isnan(out[1])


def test_incomplete_stored_snapshot_explicitly_accessible_without_status_claim():
    class Stored(Provider):
        def metadata(self):m=super().metadata();m['descriptors'][0]['availability']='stored';return m
    p=Stored();token=Event()
    with open_memory_results(p) as r:
        before=r.case_metadata('failed')
        with pytest.raises(FailedCaseError):r.read_field('failed','stress','xx')
        actual=r.read_stored_field('failed','stress','xx',[1,0,1],reduction='max_abs')
        assert actual.tolist()==[1e9,1e9,1e9] and actual.dtype==np.float64
        with pytest.raises(ValueError):actual.setflags(write=True)
        assert r.case_metadata('failed')==before and before['status']=='failed'
        assert before['diagnostics']['why']=='retained source refusal'
        token.set()
        with pytest.raises(RequestCancelled):r.read_stored_field('failed','stress','xx',cancel_event=token)
        assert len(p.calls)==1
        with pytest.raises(FailedCaseError):r.envelope('stress','xx',case_ids=['failed'])

@pytest.mark.parametrize('availability',['reconstructable','unavailable',None])
def test_stored_only_refuses_other_routes_before_any_provider_recovery(availability):
    class Other(Provider):
        descriptor_calls=0
        def metadata(self):m=super().metadata();m['descriptors'][0]['availability']=availability;return m
        def field_descriptor(self,*args):self.descriptor_calls+=1;raise AssertionError('must not recover')
    p=Other()
    with open_memory_results(p) as r:
        with pytest.raises(UnavailableFieldError):r.read_stored_field('failed','stress','xx')
        assert not p.calls and p.descriptor_calls==0


@pytest.mark.parametrize('stored_only',[False,True])
def test_cancellation_after_dynamic_descriptor_stops_before_field_recovery(stored_only):
    token=Event()
    class CancelledDescriptor(Provider):
        def metadata(self):m=super().metadata();m['descriptors'][0]['availability']='stored';return m
        def field_descriptor(self,*args):token.set();return {}
    p=CancelledDescriptor()
    with open_memory_results(p) as r:
        read=r.read_stored_field if stored_only else r.read_field
        with pytest.raises(RequestCancelled):read('failed' if stored_only else 'a','stress','xx',cancel_event=token)
    assert not p.calls


@pytest.mark.parametrize('mode',['min','max','max_abs'])
@pytest.mark.parametrize('order',[['a','b'],['b','a']])
def test_envelope_nan_policy_is_case_order_independent(mode,order):
    p=Provider();p.rows['a'][0]=[np.nan,0.,0.];p.rows['b'][0]=[7.,0.,0.]
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[0],case_ids=order,mode=mode,reduction='max',nan_policy='propagate')
        assert np.isnan(result.values[0]) and result.case_ids.tolist()==['']
        omitted=r.read_envelope('stress','xx',[0],case_ids=order,mode=mode,reduction=None,nan_policy='omit')
        assert omitted.values[0,0]==7. and omitted.case_ids[0,0]=='b'
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',[0],case_ids=order,mode=mode,reduction=None,nan_policy='raise')


def test_stored_ieee_display_snapshot_does_not_weaken_engineering_queries():
    class Stored(Provider):
        def metadata(self):m=super().metadata();m['descriptors'][0]['availability']='stored';return m
    p=Stored();p.rows['a'][0,0]=np.inf;p.rows['failed']=p.rows['a'].copy()
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.read_field('a','stress','xx')
        stored=r.read_stored_field('failed','stress','xx',[0],reduction='max_abs')
        assert stored.tolist()==[np.inf] and r.case_metadata('failed')['status']=='failed'
        with pytest.raises(ValueError):stored.setflags(write=True)
        token=Event();token.set()
        with pytest.raises(RequestCancelled):r.read_stored_field('failed','stress','xx',cancel_event=token)
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',case_ids=['a'])


def test_negative_remaining_reduction_budget_is_typed_resource_refusal():
    with pytest.raises(WorkingSetError):reduce_samples(np.array([[1.]]),working_bytes=-1)


class RaggedEnvelopeProvider(Provider):
    def __init__(self,shape=(3,)):
        super().__init__();self.shape=shape
        self.rows={'a':np.array([[1.,np.nan,np.nan],[9.,8.,np.nan],[np.nan,np.nan,np.nan]]).reshape((3,)+shape),
                   'b':np.array([[2.,7.,-7.],[5.,np.nan,np.nan],[np.nan,6.,np.nan]]).reshape((3,)+shape)}
    def metadata(self):
        m=super().metadata();m['descriptors'][0]['value_shape']=self.shape;return m
    def field_descriptor(self,cid,q,c):
        return {'value_shape':self.shape,'sample_counts':np.array([1,2,0] if cid=='a' else [3,1,2]),'present':np.array([True,True,True])}


@pytest.mark.parametrize('mode',['min','max','max_abs'])
@pytest.mark.parametrize('order',[['a','b'],['b','a']])
@pytest.mark.parametrize('shape',[(3,),(1,3)])
def test_unreduced_ragged_padding_never_becomes_source_nan(mode,order,shape):
    p=RaggedEnvelopeProvider(shape)
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[2,0,0,1],case_ids=order,mode=mode,nan_policy='propagate')
        expected=np.array([[np.nan,6.,np.nan],[1. if mode=='min' else 2.,7.,-7.],[1. if mode=='min' else 2.,7.,-7.],[5. if mode=='min' else 9.,8.,np.nan]])
        np.testing.assert_array_equal(result.values.reshape(4,3),expected)
        expected_ids=np.array([['','b',''],['a' if mode=='min' else 'b','b','b'],['a' if mode=='min' else 'b','b','b'],['b' if mode=='min' else 'a','a','']])
        np.testing.assert_array_equal(result.case_ids.reshape(4,3),expected_ids)
        # Cached values still get their own case-specific fresh presence mask.
        np.testing.assert_array_equal(r.read_envelope('stress','xx',[2,0,0,1],case_ids=order,mode=mode).values,result.values)
        assert r.diagnostics['provider_rows']==8
        token=Event();token.set()
        with pytest.raises(RequestCancelled):r.read_envelope('stress','xx',case_ids=order,mode=mode,cancel_event=token)
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',[3],case_ids=order,mode=mode)


def test_native_padding_nan_is_not_raised_or_used_as_tie_evidence():
    p=RaggedEnvelopeProvider()
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[0],case_ids=['a','b'],mode='max_abs',nan_policy='raise')
        assert result.values.tolist()==[[2.,7.,-7.]] and result.case_ids.tolist()==[['b','b','b']]
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',[2],case_ids=['b','a'],nan_policy='raise')
        assert r.read_envelope('stress','xx',[2],case_ids=['a','b'],nan_policy='omit').values.tolist()[0][1]==6.
        # Reductions keep original actual-sample semantics, including empty rows.
        assert r.read_envelope('stress','xx',[0],case_ids=['a','b'],reduction='mean',mode='max').values.tolist()==[1.]


def test_zero_selection_native_presence_is_bounded_without_ordinal_work():
    p=RaggedEnvelopeProvider()
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',np.array([],dtype=np.int64),case_ids=['a','b'])
        assert result.values.shape==(0,3) and result.case_ids.shape==(0,3)


def test_native_slot_presence_scratch_is_admitted_before_field_callback():
    class Wide(Provider):
        def field_descriptor(self,cid,q,c):return {'value_shape':(1024,),'sample_counts':np.array([1,1,1])}
    p=Wide()
    with open_memory_results(p) as r:
        r.working_bytes=50_000
        with pytest.raises(WorkingSetError):r.read_envelope('stress','xx',[0],case_ids=['a'])
        assert not p.calls



def test_reduced_index_key_payload_is_charged_and_evicted_with_result():
    p=Provider();selection=np.zeros(100,dtype=np.int64)
    with open_memory_results(p,cache_bytes=2000) as r:
        result=r.read_field('a','stress','xx',selection,reduction='max_abs')
        assert result.nbytes==800
        assert r.diagnostics['cache_resident_bytes']==48+800+800
        r.read_field('b','stress','xx',selection,reduction='max_abs')
        assert r.diagnostics['cache_resident_bytes']==48+800+800
        r.read_field('a','stress','xx',selection,reduction='max_abs')
        assert len(p.calls)==3
        r.resize_cache(1000)
        assert r.diagnostics['cache_resident_bytes']==48
    with open_memory_results(p,cache_bytes=1000) as r:
        r.read_field('a','stress','xx',selection,reduction='max_abs')
        r.read_field('a','stress','xx',selection,reduction='max_abs')
        assert r.diagnostics['cache_resident_bytes']==48


def test_partial_presence_reducer_budget_accounts_live_original_source(monkeypatch):
    import anypost.provider as module
    class Partial(Provider):
        def field_descriptor(self,*args):return {'present':np.array([True,True,False])}
    p=Partial();original=module.reduce_samples;observations=[]
    with open_memory_results(p,working_bytes=65536) as r:
        r.working_bytes=16384
        def guard(values,*args,**kwargs):
            source_bytes=p.rows['a'].nbytes
            output_bytes=len(p.ids)*8
            metadata_bytes=p.ids.nbytes+3*8
            assert kwargs['working_bytes']+source_bytes+output_bytes+metadata_bytes<=r.working_bytes
            observations.append(values.nbytes)
            return original(values,*args,**kwargs)
        monkeypatch.setattr(module,'reduce_samples',guard)
        value=r.read_field('a','stress','xx',reduction='mean')
        assert value[0]==0. and np.isnan(value[2]) and observations==[48]


@pytest.mark.parametrize('kind',['multiple','separate','generated_collision'])
def test_ambiguous_explicit_or_generated_component_keys_are_refused(kind):
    class Keys(Provider):
        def metadata(self):
            m=super().metadata();d=m['descriptors'][0]
            if kind=='multiple':d.update(components=('xx','yy'),key='stress')
            elif kind=='separate':m['descriptors']=[dict(d,key='same'),dict(d,components=('yy',),key='same')]
            else:m['descriptors']=[dict(d,quantity_id='a:b',components=('c',)),dict(d,quantity_id='a',components=('b:c',))]
            return m
    with pytest.raises(ValidationError):open_memory_results(Keys())


def test_passthrough_declared_workspace_refuses_before_source_callback():
    class Large(Provider):
        provider_work_bytes=4<<20
        allocations=0
        def geometry(self,selection=None):
            self.allocations+=1
            return {'positions':np.zeros((1<<17,3),dtype=np.float64)}
    p=Large()
    with open_memory_results(p,working_bytes=65536) as r:
        r.working_bytes=8192
        with pytest.raises(WorkingSetError):r.geometry()
        assert p.allocations==0


def test_retained_passthrough_is_admitted_before_snapshot_copy(monkeypatch):
    import anypost.provider as module
    class Retained(Provider):
        provider_work_bytes=0
        def __init__(self):super().__init__();self.huge=np.ones(2048);self.queries=0
        def query(self):self.queries+=1;return self.huge
    p=Retained()
    with open_memory_results(p,working_bytes=65536) as r:
        r.working_bytes=8192
        original=module._snapshot
        def forbidden(value,*args,**kwargs):
            if value is p.huge:raise AssertionError('must refuse before copy')
            return original(value,*args,**kwargs)
        monkeypatch.setattr(module,'_snapshot',forbidden)
        with pytest.raises(WorkingSetError):r.query()
        assert p.queries==1


def test_optional_passthrough_preflight_is_bounded_and_cancellable():
    token=Event()
    class Preflight(Provider):
        provider_work_bytes=4096
        def __init__(self):super().__init__();self.work=1<<20;self.preflights=[];self.queries=0;self.retained={'table':np.array([1.,2.])}
        def passthrough_work_bytes(self,name,*args,**kwargs):self.preflights.append(name);return self.work
        def query(self,name):self.queries+=1;return self.retained
    p=Preflight()
    with open_memory_results(p,working_bytes=32768) as r:
        with pytest.raises(WorkingSetError):r.query('forces')
        assert p.preflights==['query'] and p.queries==0
        p.work=0
        result=r.query('forces');np.testing.assert_array_equal(result['table'],[1.,2.])
        with pytest.raises(ValueError):result['table'].setflags(write=True)
        p.passthrough_work_bytes=lambda *args,**kwargs:(token.set() or 0)
        with pytest.raises(RequestCancelled):r.query('forces',cancel_event=token)
        assert p.queries==1


def test_passthrough_missing_declaration_and_cyclic_or_repeated_alias_graph():
    p=Provider();p.query=lambda:np.ones(1)
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.query()
    class Retained(Provider):
        provider_work_bytes=0
        def query(self):return self.retained
    p=Retained();shared=np.ones(1);p.retained=[shared]*100
    with open_memory_results(p,working_bytes=65536) as r:
        r.working_bytes=8192
        with pytest.raises(WorkingSetError):r.query()
    p.retained=[];p.retained.append(p.retained)
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.query()



def test_fresh_nested_transport_reserves_source_snapshot_and_public_coexistence(monkeypatch):
    import anypost.provider as module
    class Fresh(Provider):
        provider_work_bytes=8192
        latest=None
        def query(self):
            self.latest={'nested':{'a':np.ones((32,4)),'b':np.arange(128.,dtype=np.float64)},'tags':['actual','source']}
            return self.latest
    p=Fresh();original=module._snapshot;observed=[]
    with open_memory_results(p,working_bytes=65536) as r:
        def admitted(value,*args,**kwargs):
            if value is p.latest:
                payload=sum(array.nbytes for array in p.latest['nested'].values())
                assert r.diagnostics['working_peak_bytes']>=p.provider_work_bytes+payload*2
                assert r.diagnostics['working_peak_bytes']<=r.working_bytes
                observed.append(payload)
            return original(value,*args,**kwargs)
        monkeypatch.setattr(module,'_snapshot',admitted)
        result=r.query()
        assert observed==[2048]
        p.latest['nested']['a'][:]=99.
        np.testing.assert_array_equal(result['nested']['a'],np.ones((32,4)))
        with pytest.raises(ValueError):result['nested']['b'].setflags(write=True)
        with pytest.raises(TypeError):result['nested']['extra']=1



def test_envelope_explicit_empty_raise_preserves_error_and_stops_recovery():
    from anypost import EmptySampleError
    p=RaggedEnvelopeProvider()
    with open_memory_results(p) as r:
        with pytest.raises(EmptySampleError) as error:r.envelope('stress','xx',[2],case_ids=['a','b'],reduction='mean',empty_policy='raise')
        assert isinstance(error.value,UnavailableFieldError)
        assert str(error.value)=='native sample row is empty' and error.value.unavailable_case_ids==()
        assert [call[0] for call in p.calls]==['a']


def test_private_metadata_owner_cannot_be_erased_by_public_reservation_keys():
    p=Provider()
    with open_memory_results(p) as r:
        original=r.diagnostics['external_cache_bytes']
        assert original==48
        r.set_external_cache_bytes('metadata',0)
        assert r.diagnostics['external_cache_bytes']==original
        r.set_external_cache_bytes('metadata',16)
        assert r.diagnostics['external_cache_bytes']==original+16
        r.set_external_cache_bytes('metadata',0)
        with pytest.raises(ValidationError):r.set_external_cache_bytes(r._metadata_cache_owner,0)
        class Spoof:
            def __hash__(self):return hash(r._metadata_cache_owner)
            def __eq__(self,other):return True
        r.set_external_cache_bytes(Spoof(),0)
        assert r.diagnostics['external_cache_bytes']==original
        r.read_field('a','stress','xx')
        r.resize_cache(original)
        assert r.diagnostics['cache_resident_bytes']==original


def test_static_list_vectors_and_case_overrides_normalize_once_before_selected_reads(monkeypatch):
    class Lists(Provider):
        def metadata(self):
            m=super().metadata();d=m['descriptors'][0]
            d.update(ids=[41,9,700],sample_counts=[2,2,0],present=[True,True,False])
            m['case_descriptors']={'b':{'stress:xx':{'sample_counts':[1,1,0],'present':[True,False,False]}}}
            return m
    p=Lists()
    with open_memory_results(p) as r:
        def forbidden(*args,**kwargs):raise AssertionError('normalized static layout must not repeat full validation')
        monkeypatch.setattr(r,'_validate_layout',forbidden)
        assert r.components()[0]['sample_counts'].dtype.kind=='i'
        assert r.components()[0]['present'].dtype.kind=='b'
        assert r.diagnostics['external_cache_bytes']==78
        assert r.read_field('a','stress','xx',[0],reduction='mean').tolist()==[0.]
        assert r.read_field('b','stress','xx',[0],reduction='mean').tolist()==[4.]
        assert np.isnan(r.read_field('b','stress','xx',[1],reduction='mean')[0])


def test_full_vector_validation_is_admitted_before_dynamic_list_conversion(monkeypatch):
    import anypost.provider as module
    class Lists(Provider):
        def field_descriptor(self,*args):return {'sample_counts':self.large_counts}
    p=Lists();p.large_counts=[1]*10000
    with open_memory_results(p) as r:
        r.working_bytes=65536
        original=module.np.asarray;converted=[]
        def tracked(value,*args,**kwargs):
            if isinstance(value,tuple) and len(value)==10000:converted.append(True)
            return original(value,*args,**kwargs)
        monkeypatch.setattr(module.np,'asarray',tracked)
        with pytest.raises(WorkingSetError):r.read_field('a','stress','xx',[0])
        assert converted==[] and p.calls==[]


def test_large_full_vector_validation_has_explicit_working_peak():
    class Counts(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0].update(ids=np.arange(4096),sample_counts=np.ones(4096,dtype=np.int64),present=np.ones(4096,dtype=bool))
            return m
    with open_memory_results(Counts(),working_bytes=1<<20) as r:
        assert r.diagnostics['working_peak_bytes']>=4096*2
        assert r.diagnostics['external_cache_bytes']==4096*(8+8+1)
        assert r._layout(r._descriptor('stress','xx'))[0]==4096


@pytest.mark.parametrize('case_ids',['failed',b'failed'])
def test_envelope_case_sequence_and_all_failed_coverage_are_explicit(case_ids):
    with open_memory_results(Provider()) as r:
        with pytest.raises(ValidationError):r.envelope('stress','xx',case_ids=case_ids)
        with pytest.raises(FailedCaseError) as error:r.envelope('stress','xx',case_ids=['failed'])
        assert error.value.failed_case_ids==('failed',) and error.value.requested_case_ids==('failed',)
        with pytest.raises(FailedCaseError) as error:r.envelope('stress','xx',case_ids=['a','failed'],strict_failed=True)
        assert error.value.failed_case_ids==('failed',) and error.value.requested_case_ids==('a','failed')
        result=r.read_envelope('stress','xx',[],case_ids=['a','b'])
        assert result.values.shape==(0,3)


def test_constructor_list_normalization_refuses_conservatively_under_small_cap():
    class Lists(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0].update(ids=list(range(4096)),sample_counts=[1]*4096,present=[True]*4096)
            return m
    with pytest.raises(WorkingSetError):open_memory_results(Lists(),working_bytes=32768)



def test_metadata_coexistence_is_admitted_before_snapshot_for_shared_arrays(monkeypatch):
    import anypost.provider as module
    class Shared(Provider):
        def metadata(self):
            m=super().metadata();ids=np.arange(1024,dtype=np.int64)
            m['descriptors']=[dict(quantity_id='q'+str(i),components=('x',),location='element',ids=ids,provider_work_bytes=0) for i in range(8)]
            return m
    def forbidden(*args,**kwargs):raise AssertionError('insufficient metadata budget must refuse before snapshot')
    monkeypatch.setattr(module,'_snapshot',forbidden)
    with pytest.raises(WorkingSetError):open_memory_results(Shared(),working_bytes=32768)


def test_shared_metadata_snapshot_payload_is_deduplicated_when_admitted():
    class Shared(Provider):
        def metadata(self):
            m=super().metadata();ids=np.arange(32,dtype=np.int64)
            m['descriptors']=[dict(quantity_id='q'+str(i),components=('x',),location='element',ids=ids,provider_work_bytes=0) for i in range(3)]
            return m
    with open_memory_results(Shared()) as r:
        assert r.diagnostics['external_cache_bytes']==32*8
        assert r._meta['descriptors'][0]['ids'] is r._meta['descriptors'][1]['ids']



def test_unique_metadata_graph_list_coexistence_refuses_before_snapshot(monkeypatch):
    import anypost.provider as module
    class Lists(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0]['sample_counts']=[1]*4096;return m
    def forbidden(*args,**kwargs):raise AssertionError('constructor must pre-admit unique input and snapshot coexistence')
    monkeypatch.setattr(module,'_snapshot',forbidden)
    with pytest.raises(WorkingSetError):open_memory_results(Lists(),working_bytes=50000)


def test_metadata_cycles_refuse_before_recursive_snapshot(monkeypatch):
    import anypost.provider as module
    class Cyclic(Provider):
        def metadata(self):m=super().metadata();m['cycle']=m;return m
    monkeypatch.setattr(module,'_snapshot',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('must refuse before copying')))
    with pytest.raises(ValidationError):open_memory_results(Cyclic())


@pytest.mark.parametrize('override',[{'dtype':'float32'},{'ids':[9,41,700]}, {'spatial_count':4},{'unit':'MPa'},{'basis':'world'},{'location':'node'},{'component':'yy'}])
@pytest.mark.parametrize('dynamic',[False,True])
def test_fixed_reader_axes_precision_and_component_identity_refuse_before_field(override,dynamic):
    class Changed(Provider):
        def metadata(self):
            m=super().metadata()
            if not dynamic:m['case_descriptors']={'b':{'stress:xx':override}}
            return m
        def field_descriptor(self,*args):return override if dynamic else {}
    p=Changed()
    if dynamic:
        with open_memory_results(p) as r:
            with pytest.raises(ValidationError):r.read_field('b','stress','xx')
            assert not p.calls
    else:
        with pytest.raises(ValidationError):open_memory_results(p)
        assert not p.calls

@pytest.mark.parametrize('mapping',[{'nope':{'stress:xx':{}}},{'a':{'stress:XX':{}}}])
def test_unknown_static_case_override_is_not_silently_ignored(mapping):
    class Changed(Provider):
        def metadata(self):m=super().metadata();m['case_descriptors']=mapping;return m
    with pytest.raises(ValidationError):open_memory_results(Changed())

@pytest.mark.parametrize('kind',['static','case','dynamic'])
def test_unavailable_metadata_prevents_field_recovery_and_reports_envelope_case(kind):
    class Missing(Provider):
        def metadata(self):
            m=super().metadata()
            if kind=='static':m['descriptors'][0]['availability']='unavailable'
            if kind=='case':m['case_descriptors']={'a':{'stress:xx':{'availability':'unavailable'}}}
            return m
        def field_descriptor(self,*args):return {'availability':'unavailable'} if kind=='dynamic' else {}
    p=Missing()
    with open_memory_results(p) as r:
        with pytest.raises(UnavailableFieldError) as error:r.read_field('a','stress','xx')
        assert error.value.unavailable_case_ids==('a',)
        with pytest.raises(UnavailableFieldError) as error:r.read_envelope('stress','xx',case_ids=['a'])
        assert error.value.unavailable_case_ids==('a',) and not p.calls

@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32])
def test_narrow_selector_preallocation_is_admitted_in_int64_domain(dtype):
    p=Provider()
    with open_memory_results(p) as r:
        r.working_bytes=12000
        with pytest.raises(WorkingSetError):r.read_field('a','stress','xx',np.zeros(1000,dtype=dtype))
        assert not p.calls

@pytest.mark.parametrize('selection',[[0]*1000,tuple([0]*1000)])
def test_python_selector_allocation_refuses_before_asarray(selection,monkeypatch):
    import anypost.provider as module
    with open_memory_results(Provider()) as r:
        r.working_bytes=12000
        def forbidden(*args,**kwargs):raise AssertionError('selector allocation before admission')
        monkeypatch.setattr(module.np,'asarray',forbidden)
        with pytest.raises(WorkingSetError):r._select(3,selection)

@pytest.mark.parametrize('kwargs',[{'reduction':'avg'},{'nan_policy':'skip'},{'empty_policy':'silent'},{'reduction':'magnitude'}])
def test_invalid_query_policies_fail_before_field_recovery(kwargs):
    p=Provider()
    with open_memory_results(p) as r:
        with pytest.raises((ValueError,ValidationError)):r.read_field('a','stress','xx',**kwargs)
        assert not p.calls

def test_raw_nan_raise_only_actual_native_slots_and_raw_empty_policy_documented():
    p=Provider()
    with open_memory_results(p) as r:
        with pytest.raises(ValidationError):r.read_field('a','stress','xx',[1],nan_policy='raise')
        # Zero samples are absent from actual evidence; raw data remains raw.
        result=r.read_field('a','stress','xx',[2],nan_policy='raise',empty_policy='raise')
        assert result.tolist()==[[0.,77.,66.]]

@pytest.mark.parametrize('reduction',[None,'mean'])
def test_engineering_infinity_admission_ignores_only_explicit_padding_and_absence(reduction):
    class Sparse(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0]['present']=np.array([True,False,True]);return m
    p=Sparse();p.rows['a'][0,2]=np.inf;p.rows['a'][1,:]=np.inf;p.rows['a'][2,:]=np.inf
    with open_memory_results(p) as r:
        v=r.read_field('a','stress','xx',reduction=reduction)
        if reduction is None:assert np.isinf(v[0,2]) and np.isnan(v[1]).all()
        else:assert v[0]==0 and np.isnan(v[1]) and np.isnan(v[2])
        r.clear_cache();p.rows['a'][0,0]=np.inf
        with pytest.raises(ValidationError):r.read_field('a','stress','xx',reduction=reduction)

@pytest.mark.parametrize('bad',[[],[[]],[None],[1]])
def test_envelope_request_shape_is_typed_not_failed_case(bad):
    with open_memory_results(Provider()) as r:
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',case_ids=bad)

def test_single_failed_case_error_has_exact_coverage():
    with open_memory_results(Provider()) as r:
        with pytest.raises(FailedCaseError) as e:r.read_field('failed','stress','xx')
        assert e.value.failed_case_ids==('failed',) and e.value.requested_case_ids==('failed',)

def test_dynamic_normalization_uses_cached_metadata_sizes_not_whole_case_walk(monkeypatch):
    import anypost.provider as module
    class Dynamic(Provider):
        def field_descriptor(self,*args):return {'sample_counts':[1,1,0]}
    with open_memory_results(Dynamic()) as r:
        old_size=module._size;old_arrays=module._arrays
        def size(value,*args):
            assert value is not r._meta,'whole metadata walk on query'
            return old_size(value,*args)
        def arrays(value,*args):
            assert value is not r._meta,'whole metadata payload walk on query'
            return old_arrays(value,*args)
        monkeypatch.setattr(module,'_size',size);monkeypatch.setattr(module,'_arrays',arrays)
        assert r.read_field('a','stress','xx',[0],reduction='mean').tolist()==[-2.]
        assert r.read_field('b','stress','xx',[0],reduction='mean').tolist()==[4.]

def test_case_shapes_counts_can_change_without_changing_fixed_entity_axes():
    class Dynamic(Provider):
        def field_descriptor(self,cid,*args):return {'ids':self.ids.copy(),'value_shape':(2,),'sample_counts':[2,1,0]}
        def read_field(self,cid,q,c,selection=None,cancel_event=None):return super().read_field(cid,q,c,selection,cancel_event)[:,:2]
    with open_memory_results(Dynamic()) as r:
        assert r.read_field('a','stress','xx',[0],reduction='max_abs').tolist()==[-2.]


@pytest.mark.parametrize('unavailable',[['stress:XX'],'stress:xx',5,['stress:xx','stress:xx'],[None]])
def test_case_unavailable_component_vocabulary_is_admitted_before_queries(unavailable):
    class Changed(Provider):
        def metadata(self):m=super().metadata();m['cases'][0]['unavailable_components']=unavailable;return m
    p=Changed()
    with pytest.raises(ValidationError):open_memory_results(p)
    assert not p.calls

def test_known_case_unavailable_components_preserve_case_identity_and_no_recovery():
    class Changed(Provider):
        def metadata(self):m=super().metadata();m['cases'][0]['unavailable_components']=['stress:xx'];return m
    p=Changed()
    with open_memory_results(p) as r:
        with pytest.raises(UnavailableFieldError) as e:r.read_field('a','stress','xx')
        assert e.value.unavailable_case_ids==('a',) and not p.calls

@pytest.mark.parametrize('count',[4,'3',True,3.0,-1,None])
def test_explicit_spatial_count_must_match_ids_with_integer_type(count):
    class Changed(Provider):
        def metadata(self):m=super().metadata();m['descriptors'][0]['spatial_count']=count;return m
    p=Changed()
    with pytest.raises(ValidationError):open_memory_results(p)
    assert not p.calls

def test_exact_spatial_count_and_shape_aliases_are_preserved_with_extension_metadata():
    class Changed(Provider):
        def metadata(self):
            m=super().metadata();m['descriptors'][0].update(spatial_count=3,sample_shape=(3,),application_extension={'notes':'opaque provenance'});return m
    with open_memory_results(Changed()) as r:
        assert r.components()[0]['application_extension']['notes']=='opaque provenance'
        assert r.read_field('a','stress','xx',[0]).tolist()==[[-2.,2.,99.]]

@pytest.mark.parametrize('kind',['static','case','dynamic'])
def test_conflicting_sample_shape_aliases_refuse_before_field_recovery(kind):
    class Changed(Provider):
        def metadata(self):
            m=super().metadata()
            if kind=='static':m['descriptors'][0]['sample_shape']=(2,)
            if kind=='case':m['case_descriptors']={'a':{'stress:xx':{'sample_shape':(2,)}}}
            return m
        def field_descriptor(self,*args):return {'sample_shape':(2,)} if kind=='dynamic' else {}
    p=Changed()
    if kind=='dynamic':
        with open_memory_results(p) as r:
            with pytest.raises(ValidationError):r.read_field('a','stress','xx')
    else:
        with pytest.raises(ValidationError):open_memory_results(p)
    assert not p.calls

class EmptyChangingProvider(Provider):
    def field_descriptor(self,cid,*args):return {'sample_counts':np.array([2,2,0 if cid=='a' else 1])}

@pytest.mark.parametrize('mode',['min','max','max_abs'])
@pytest.mark.parametrize('policy',['propagate','raise','omit'])
@pytest.mark.parametrize('order',[['a','b'],['b','a']])
def test_reduced_zero_samples_are_missing_evidence_not_actual_source_nan(mode,policy,order):
    with open_memory_results(EmptyChangingProvider()) as r:
        field=r.read_field('a','stress','xx',[2],reduction='mean',nan_policy=policy)
        assert np.isnan(field[0])
        # Field was cached first: envelope must still compute correct evidence.
        result=r.read_envelope('stress','xx',[2],case_ids=order,mode=mode,reduction='mean',nan_policy=policy)
        assert result.values.tolist()==[0.] and result.case_ids.tolist()==['b']
        assert result.admitted_case_ids==tuple(order)

@pytest.mark.parametrize('mode',['min','max','max_abs'])
def test_explicit_empty_zero_keeps_governing_ties_and_raise_still_refuses(mode):
    from anypost import EmptySampleError
    with open_memory_results(EmptyChangingProvider()) as r:
        result=r.read_envelope('stress','xx',[2],case_ids=['a','b'],mode=mode,reduction='mean',empty_policy='zero',nan_policy='raise')
        assert result.values.tolist()==[0.] and result.case_ids.tolist()==['a']
        with pytest.raises(EmptySampleError):r.read_envelope('stress','xx',[2],case_ids=['a','b'],mode=mode,reduction='mean',empty_policy='raise')

@pytest.mark.parametrize('order',[['a','b'],['b','a']])
def test_empty_evidence_does_not_hide_actual_source_nan(order):
    p=EmptyChangingProvider();p.rows['b'][2,0]=np.nan
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[2],case_ids=order,reduction='mean')
        assert np.isnan(result.values[0]) and result.case_ids.tolist()==['']
        with pytest.raises(ValidationError):r.read_envelope('stress','xx',[2],case_ids=order,reduction='mean',nan_policy='raise')

def test_all_empty_reduced_cases_have_no_governing_identity_or_source_nan_error():
    class Empty(Provider):
        def field_descriptor(self,*args):return {'sample_counts':[2,2,0]}
    with open_memory_results(Empty()) as r:
        result=r.read_envelope('stress','xx',[2],case_ids=['a','b'],reduction='mean',nan_policy='raise')
        assert np.isnan(result.values[0]) and result.case_ids.tolist()==['']

def test_field_descriptor_missing_case_field_has_unavailable_coverage():
    class Missing(Provider):
        def field_descriptor(self,*args):raise KeyError('no actual source field')
    p=Missing()
    with open_memory_results(p) as r:
        with pytest.raises(UnavailableFieldError) as e:r.read_envelope('stress','xx',case_ids=['a','b'],reduction='mean')
        assert e.value.unavailable_case_ids==('a','b') and not p.calls
        assert isinstance(e.value.__cause__,type(None))

@pytest.mark.parametrize('dynamic',[False,True])
def test_sampling_definition_is_fixed_but_case_provenance_remains_extensible(dynamic):
    class Changed(Provider):
        def metadata(self):
            m=super().metadata()
            if not dynamic:m['case_descriptors']={'a':{'stress:xx':{'sampling':'averaged','provenance':{'case_detail':'actual'}}}}
            return m
        def field_descriptor(self,*args):return {'sampling':'averaged'} if dynamic else {}
    p=Changed()
    if dynamic:
        with open_memory_results(p) as r:
            with pytest.raises(ValidationError):r.read_field('a','stress','xx')
    else:
        with pytest.raises(ValidationError):open_memory_results(p)
    assert not p.calls


@pytest.mark.parametrize('bad_ids',[5,3.0,'41,9,700',{'first':41},True])
@pytest.mark.parametrize('dynamic',[False,True])
def test_nonvector_ids_refuse_even_with_valid_spatial_count(bad_ids,dynamic):
    class Bad(Provider):
        def metadata(self):
            m=super().metadata()
            if not dynamic:m['descriptors'][0].update(ids=bad_ids,spatial_count=3)
            return m
        def field_descriptor(self,*args):return {'ids':bad_ids,'spatial_count':3} if dynamic else {}
    p=Bad()
    if dynamic:
        with open_memory_results(p) as r:
            with pytest.raises(ValidationError):r.read_field('a','stress','xx')
    else:
        with pytest.raises(ValidationError):open_memory_results(p)
    assert not p.calls

@pytest.mark.parametrize('method',['mean','min','max','max_abs'])
@pytest.mark.parametrize('empty_policy',['zero','raise'])
def test_omit_all_nan_counted_samples_are_not_structural_empty(method,empty_policy):
    v=np.array([[np.nan,np.nan],[np.nan,900.],[2.,4.]])
    counts=np.array([2,1,2])
    out=reduce_samples(v,method,sample_counts=counts,nan_policy='omit',empty_policy=empty_policy)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2])

@pytest.mark.parametrize('empty_policy',['zero','raise'])
@pytest.mark.parametrize('order',[['a','b'],['b','a']])
def test_omit_all_nan_case_does_not_hide_finite_governing_case(empty_policy,order):
    p=Provider();p.rows['a'][0,:2]=np.nan
    with open_memory_results(p) as r:
        result=r.read_envelope('stress','xx',[0],case_ids=order,reduction='max_abs',nan_policy='omit',empty_policy=empty_policy)
        assert result.values.tolist()==[4.] and result.case_ids.tolist()==['b']

@pytest.mark.parametrize('policy',['zero','raise'])
def test_structural_empty_policy_remains_distinct_from_all_nan_omit(policy):
    from anypost import EmptySampleError
    v=np.array([[np.nan,np.nan],[7.,8.]])
    counts=np.array([2,0])
    if policy=='raise':
        with pytest.raises(EmptySampleError):reduce_samples(v,'mean',sample_counts=counts,nan_policy='omit',empty_policy=policy)
    else:
        out=reduce_samples(v,'mean',sample_counts=counts,nan_policy='omit',empty_policy=policy)
        assert np.isnan(out[0]) and out[1]==0.

def test_descriptor_keyerror_direct_cause_and_unknown_case_preserved():
    class Missing(Provider):
        def field_descriptor(self,*args):raise KeyError('source field missing')
    p=Missing()
    with open_memory_results(p) as r:
        with pytest.raises(UnavailableFieldError) as e:r.read_field('a','stress','xx')
        assert isinstance(e.value.__cause__,KeyError) and e.value.unavailable_case_ids==('a',)
        with pytest.raises(UnknownCaseError):r.read_field('never','stress','xx')
        assert not p.calls
