"""Installed runtime2/source record integration; intentionally skipped by runtime1."""
import numpy as np
import pytest
import anysolver_results as runtime
from anypost import write_results,open_results,ArchiveSink,UnavailableFieldError
from test_stored_correction import bits_equal,capsule
pytestmark=pytest.mark.skipif(runtime.RUNTIME_ID!='anysolver-results/2',reason='Requires frozen source/runtime2 installed pair')


def source_values(record,key):
    if key.startswith('principal_stress_'):
        rank,surface=key.split(':');rank=int(rank.removeprefix('principal_stress_'))
        return np.concatenate([np.asarray(value['principal_stresses_'+surface])[:,rank-1] for _,value in sorted(record['element_stresses'].items())])
    if key.startswith('displacement:'):
        column=['UX','UY','UZ','RX','RY','RZ'].index(key.split(':')[1])
        return np.array([value[column] for _,value in sorted(record['node_displacements'].items())])
    if key.startswith('reaction:'):
        column=['FX','FY','FZ','MX','MY','MZ'].index(key.split(':')[1])
        return np.array([value[column] for _,value in sorted(record['reactions'].items())])
    from test_result_capsule_export import _reference_fields
    return np.concatenate([value for _,value in sorted(_reference_fields(record)[key].items())])


@pytest.mark.parametrize('fixture',['beam2','beam3','rotated_shell_pressure'])
def test_actual_source_records_writer_stream_selectors_and_envelope_bits(tmp_path,fixture):
    from anysolver import PatternCase,RecoveryConfig,FixedSupport,BoundaryCondition,prepare_static_response_evaluator,export_recovery_capsule,export_recovery_capsule_to_sink
    from test_prepared_static_evaluator import _bar,_case,_shell_case
    from test_result_capsule_export import _rotated_shell_model
    if fixture=='rotated_shell_pressure':
        model=_rotated_shell_model();case_factory=lambda i,v:_shell_case(i,pressure=v)
        components=['membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz','von_mises','principal_stresses_top','principal_stresses_bottom']
    else:
        if fixture=='beam3':
            from test_generalized_beam_sections import _model,_stiffness
            from anysolver.elements import QuadraticBeamElement
            from anysolver.beam_sections import GeneralizedBeamSection
            model,_=_model(QuadraticBeamElement,section=GeneralizedBeamSection(_stiffness()))
            model.add_boundary_condition(FixedSupport('base',[1]))
            model.add_boundary_condition(BoundaryCondition('axial',[2,3],{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
        else:model=_bar()
        case_factory=_case;components=None
    patterns={f'p{i}':case_factory(i+1,(i+1)*.7).load_case for i in range(4)}
    baseline=case_factory(0,.5).load_case
    dense={'p3':2.1,'p0':-.9,'p1':.3,'p2':1.7}
    cases=[PatternCase('dense',dense),PatternCase('sparse',{'p3':2.2,'p0':-.9}),PatternCase('empty',{}),PatternCase('tie',dense),PatternCase('canonical',{'p0':1.},canonical_load_case=case_factory(7,5.).load_case)]
    config=RecoveryConfig(include_stresses=True,include_reactions=True,components=components)
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,recovery_config=config,stress_execution='certified_library') as evaluator:
        records=next(evaluator.iter_batches(cases)).case_results
        c=export_recovery_capsule(evaluator,cases,model_identity=fixture,mesh_identity='actual-records')
        assert c['version']==2 and c['cases'][-1]['portable'] is False
        ordinary=write_results(tmp_path/'whole.h5',c)
        streamed=tmp_path/'stream.h5'
        with ArchiveSink(streamed) as sink:export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity=fixture,mesh_identity='actual-records',batch_size=2)
        legacy=export_recovery_capsule(evaluator,cases,model_identity=fixture,mesh_identity='actual-records',replay_version=1)
        legacy_path=write_results(tmp_path/'legacy.h5',legacy)
    for path in (ordinary,streamed):
        with open_results(path,working_bytes=8192,cache_bytes=4096) as reader:
            assert reader.cases()==tuple(c['cases'])
            for index,record in enumerate(records):
                for d in c['descriptors']:
                    key=d['key']
                    if key in c['cases'][index].get('unavailable_components',()) or d['availability']=='unavailable':
                        with pytest.raises(UnavailableFieldError):reader.read_field(cases[index].case_id,d['quantity_id'],d['component'],[])
                        continue
                    expected=source_values(record,key);n=len(expected)
                    for selection in (None,slice(None,None,-1),np.array([n-1,0,n-1]),np.arange(0,n,2)):
                        chosen=np.arange(n) if selection is None else np.arange(n)[selection]
                        bits_equal(reader.read_field(cases[index].case_id,d['quantity_id'],d['component'],selection),expected[chosen])
                    if key.startswith('principal_stress_'):assert d['availability']=='stored' and key not in c['derived']
                    if index==0:
                        envelope=reader.read_envelope(d['quantity_id'],d['component'],case_ids=['tie','dense'],selection=[0,0])
                        assert envelope.min_case_ids.tolist()==envelope.max_case_ids.tolist()==['dense','dense']
            assert reader.diagnostics['working_peak_bytes']<=8192 and reader.diagnostics['cache_peak_bytes']<=4096
    with open_results(legacy_path,working_bytes=65536) as reader:
        # New runtime advertises and explicitly dispatches the old BLAS contract.
        key=next(k for k in legacy['responses'] if not k.startswith('reaction:') and 0 not in legacy.get('stored',{}).get(k,{}).get('case_indices',()))
        d=next(d for d in legacy['descriptors'] if d['key']==key)
        value=reader.read_field('dense',d['quantity_id'],d['component'],[0])
        assert value.shape==(1,) and reader._runtime.RUNTIME_ID=='anysolver-results/2'


def test_actual_ordered_runtime_tiled_sparse_duplicates_and_prepared_accounting(tmp_path):
    c=dict(capsule(1400,native=False),version=2,runtime='anysolver-results/2',combination='ordered_float64/1',stored={})
    c['cases']=tuple(dict(case,portable=case['certified'],displacement_portable=True if case['certified'] else None) for case in c['cases'])
    path=write_results(tmp_path/'ordered.h5',c)
    with open_results(path,working_bytes=32768,cache_bytes=8192) as reader:
        for index in range(4):
            active=c['cases'][index]['active_indices'];weights=c['coefficients'][index] if active is None else c['coefficients'][index][np.asarray(active,dtype=int)-1]
            expected=runtime.reconstruct_response(c['responses']['sx'],weights,active_indices=active,combination='ordered_float64/1')
            for selection in (None,np.array([1399,0,1399,7]),slice(None,None,-1)):
                chosen=np.arange(1400) if selection is None else np.arange(1400)[selection]
                bits_equal(reader.read_field(f'c{index}','stress','x',selection),expected[chosen])
        prepared=[v for key,(v,size) in reader._cache._values.items() if key[0]=='prepared_response']
        assert prepared and all(p.combination=='ordered_float64/1' and p.nbytes==p.baseline.nbytes+p.selected.nbytes for p in prepared)
        assert reader.diagnostics['working_peak_bytes']<=32768 and reader.diagnostics['cache_peak_bytes']<=8192
