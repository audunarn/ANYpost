"""Versioned lossless byte prediction, never an engineering recovery path.

Every stored tile carries the SHA256 of the original float64 words. A decoder
must reproduce those words or refuse before returning any selected value.
"""
from collections.abc import Mapping
import hashlib
import json
import importlib
import numpy as np
from .errors import ValidationError, RuntimeDependencyError, WorkingSetError
from .validation import array, json_text
from .stored_codec import predictor_bits

ROLE = 'storage_prediction/1'
CODEC = 'xor_storage_prediction_sha256/1'
ROWS = 256 * 1024 // 8
WORK_BYTES = 128 * 1024**2
_INPUTS = {
    'beam_von_mises': ('axial_stress','bending_stress_y','bending_stress_z','shear_stress_y','shear_stress_z','torsional_stress'),
    'shell_von_mises': ('membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz'),
    'equivalent_stress': ('von_mises',),
}

class PredictionOverflow(ArithmeticError):
    """Only validated numeric prediction overflow permits a durable zero mode."""

def words_digest(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes(order='C')).hexdigest()

def associations_digest(values):
    digest=hashlib.sha256()
    for first in range(0,len(values),ROWS):digest.update(np.asarray(values[first:first+ROWS],dtype='<i8').tobytes(order='C'))
    return digest.hexdigest()

def contract(role):
    if not isinstance(role,Mapping) or type(role.get('version')) is not int or role['version'] != 1:
        raise ValidationError('unknown storage prediction role/version')
    if type(role.get('runtime')) is not str or type(role.get('combination')) is not str or role['runtime'] != 'anysolver-results/2' or role['combination'] != 'ordered_float64/1':
        raise ValidationError('storage prediction runtime/combination mismatch')
    primitives=role.get('responses');derived=role.get('derived');targets=role.get('targets')
    if not isinstance(primitives,(Mapping,list,tuple)) or not isinstance(derived,Mapping) or not isinstance(targets,(list,tuple)):
        raise ValidationError('storage prediction requires primitive/derived/target tables')
    if any(not isinstance(k,str) or not k for k in primitives) or any(not isinstance(k,str) or not k for k in derived) or any(not isinstance(k,str) or not k for k in targets):
        raise ValidationError('storage predictor keys must be nonempty strings')
    keys=set(primitives)|set(derived)
    if len(set(primitives))!=len(primitives) or set(primitives)&set(derived) or not keys:
        raise ValidationError('storage predictor keys overlap or are empty')
    if any(not isinstance(k,str) or not k for k in keys) or len(set(targets))!=len(targets) or not set(targets)<=keys:
        raise ValidationError('invalid storage predictor target keys')
    for key,spec in derived.items():
        if not isinstance(spec,Mapping) or set(spec)!={'operation','inputs'} or type(spec.get('operation')) is not str or spec['operation'] not in _INPUTS:
            raise ValidationError('unsupported storage derived rule; principal predictors remain plain')
        inputs=spec['inputs']
        if not isinstance(inputs,Mapping) or set(inputs)!=set(_INPUTS[spec['operation']]) or any(not isinstance(v,str) or v not in keys for v in inputs.values()):
            raise ValidationError('storage derived predictor has missing or malformed inputs')
    def visit(key,path):
        if key in path:raise ValidationError('cyclic storage predictor dependency')
        for dep in derived.get(key,{}).get('inputs',{}).values():visit(dep,path|{key})
    for key in keys:visit(key,set())
    return keys

def manifest(role,patterns,associations=None,descriptors=None):
    """Admit source arrays, returning compact JSON without retaining copies."""
    keys=contract(role)
    allowed={'version','runtime','combination','responses','derived','targets','associations','provenance'}
    if set(role)-allowed:raise ValidationError('unknown storage prediction role fields')
    if not isinstance(role['responses'],Mapping) or not isinstance(role.get('associations'),Mapping) or set(role['associations'])!=keys:
        raise ValidationError('storage predictors require exact association tables')
    if descriptors is not None:
        declared={d['key']:d for d in descriptors}
        if not keys<=declared.keys():raise ValidationError('storage prediction refers to missing descriptor namespaces')
        if any(declared[k]['availability']!='stored' for k in role['targets']):raise ValidationError('storage-only target must be scientifically stored')
    counts={};proofs={}
    for key in keys:
        ids=array(role['associations'][key],f'storage predictor associations/{key}',integer=True,ndim=1)
        counts[key]=len(ids);proofs[key]=associations_digest(ids)
        if associations is not None:
            if key not in associations:raise ValidationError('storage prediction has no source association namespace')
            if not np.array_equal(ids,np.asarray(associations[key])):raise ValidationError('storage prediction/source association mismatch')
    for key,values in role['responses'].items():
        values=array(values,f'storage predictor/{key}',ndim=2)
        if values.dtype!=np.dtype('float64') or values.shape!=(counts[key],len(patterns)+1):
            raise ValidationError('storage primitive predictor must preserve full float64 row/pattern layout')
    for key,spec in role['derived'].items():
        if any(counts[key]!=counts[dep] or proofs[key]!=proofs[dep] for dep in spec['inputs'].values()):
            raise ValidationError('storage derived predictor associations differ')
    return dict(version=1,runtime=role['runtime'],combination=role['combination'],responses=sorted(role['responses']),derived=json.loads(json_text(role['derived'])),targets=list(role['targets']),counts=counts,association_sha256=proofs,provenance=json.loads(json_text(role.get('provenance',{}))))

def validate_manifest(role,patterns):
    keys=contract(role)
    if set(role)!={'version','runtime','combination','responses','derived','targets','counts','association_sha256','provenance'}:
        raise ValidationError('invalid persisted storage prediction role schema')
    if not isinstance(role['responses'],list) or not isinstance(role['counts'],Mapping) or not isinstance(role['association_sha256'],Mapping) or set(role['counts'])!=keys or set(role['association_sha256'])!=keys:
        raise ValidationError('invalid persisted storage predictor counts/proofs')
    for key in keys:
        count=role['counts'][key];digest=role['association_sha256'][key]
        if type(count) is not int or count<0 or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValidationError('invalid storage predictor layout proof')
    for key,spec in role['derived'].items():
        if any(role['counts'][key]!=role['counts'][dep] or role['association_sha256'][key]!=role['association_sha256'][dep] for dep in spec['inputs'].values()):
            raise ValidationError('storage prediction dependency association mismatch')
    return keys

def install(handle,role,patterns,groups,dataset,compression):
    info=manifest(role,patterns)
    root=handle.create_group('storage_prediction')
    root.attrs['role']=ROLE
    for key in sorted(contract(role)):
        group=root.create_group(groups(key))
        group.attrs['key']=key
        dataset(group,'associations',role['associations'][key],compression,integer=True)
        if key in role['responses']:dataset(group,'responses',role['responses'][key],compression)
    return info

def validate_disk(handle,info,patterns,groups,descriptors,counts,check=None,on_work=None):
    """Metadata-only admission: never import the optional runtime."""
    root=handle.get('storage_prediction')
    if info is None:
        if root is not None:raise ValidationError('stray storage predictor datasets without role metadata')
        return
    keys=validate_manifest(info,patterns)
    if not keys<=groups.keys() or not keys<=counts.keys() or not keys<=descriptors.keys():
        raise ValidationError('storage prediction has missing source namespace/count')
    if root is None or root.attrs.get('role')!=ROLE or set(root)!=set(groups[key] for key in keys):
        raise ValidationError('storage prediction role/datasets disagree')
    for key in keys:
        if key not in descriptors or counts[key]!=info['counts'][key]:raise ValidationError('storage prediction namespace/source count mismatch')
        group=root[groups[key]]
        if group.attrs.get('key')!=key or set(group)!=( {'associations','responses'} if key in info['responses'] else {'associations'}):
            raise ValidationError('stray/missing storage predictor datasets')
        ids=group['associations'];actual=handle['components'][groups[key]]['associations']
        if ids.shape!=actual.shape or ids.ndim!=1 or ids.dtype!=np.dtype('int64'):
            raise ValidationError('storage predictor associations have invalid layout')
        digest=hashlib.sha256()
        for first in range(0,len(ids),ROWS):
            if check is not None:check()
            if on_work is not None:on_work(min(ROWS,len(ids)-first)*32)
            old=ids[first:first+ROWS];new=actual[first:first+ROWS]
            if not np.array_equal(old,new):raise ValidationError('storage predictor associations differ from actual source')
            digest.update(np.asarray(old,dtype='<i8').tobytes())
        if digest.hexdigest()!=info['association_sha256'][key]:raise ValidationError('storage predictor association digest mismatch')
        if key in info['responses']:
            ds=group['responses']
            if ds.dtype!=np.dtype('float64') or ds.shape!=(counts[key],len(patterns)+1):raise ValidationError('storage predictor response shape mismatch')
    for key in info['targets']:
        if descriptors[key]['availability']!='stored' or 'responses' in handle['components'][groups[key]]:
            raise ValidationError('storage-only predictor escaped into scientific recovery availability')

def closure(info,key):
    found=set()
    def visit(k):
        if k in found:return
        found.add(k)
        for dep in info['derived'].get(k,{}).get('inputs',{}).values():visit(dep)
    visit(key);return found

def primitives(info,key):
    return sorted(closure(info,key)&set(info['responses']))

def work_bytes(info,key,rows,width):
    # All dependency bases stay live together: retained source matrices plus
    # conservative conversion/load scratch, finite checks and derived arrays.
    return rows*(16*width*len(primitives(info,key))+128*len(closure(info,key))+256)+32*width

def tile_rows(info,key,count,width):
    fixed=work_bytes(info,key,0,width);per_row=work_bytes(info,key,1,width)-fixed
    if fixed+per_row>WORK_BYTES:raise WorkingSetError('even one storage integrity row exceeds128MiB workspace')
    # Leave headroom for selected output and index buffers on the read path.
    capacity=max(1,(WORK_BYTES//2-fixed)//per_row)
    return max(1,min(count,ROWS,capacity))

def admit_basis(values,rows,width):
    values=np.asarray(values)
    if values.dtype!=np.dtype('float64') or values.shape!=(rows,width) or not np.isfinite(values).all():
        raise ValidationError('malformed/nonfinite storage predictor basis')
    values.setflags(write=False)
    return values

def runtime():
    try:module=importlib.import_module('anysolver_results')
    except ImportError as error:raise RuntimeDependencyError('derived storage decoding needs ANYpost[results] runtime2') from error
    if getattr(module,'RUNTIME_ID',None)!='anysolver-results/2' or 'ordered_float64/1' not in getattr(module,'SUPPORTED_COMBINATIONS',()) or type(getattr(module,'STORAGE_PREDICTION_API',None)) is not int or module.STORAGE_PREDICTION_API!=1 or not callable(getattr(module,'validate_storage_prediction',None)):
        raise RuntimeDependencyError('derived storage predictor needs runtime2 ordered_float64/1')
    return module

def predict(info,key,first,stop,weights,active,basis,check=None,*,basis_verified=False):
    memo={}
    def evaluate(k):
        if check is not None:check()
        if k in memo:return memo[k]
        if k in info['responses']:
            source=np.asarray(basis(k,first,stop),dtype=np.float64)
            if source.shape!=(stop-first,len(weights)+1) or (not basis_verified and not np.isfinite(source).all()):
                raise ValidationError('malformed/nonfinite storage predictor basis')
            result=predictor_bits(source,weights,active).view(np.float64)
        else:
            spec=info['derived'][k]
            inputs={name:evaluate(dep) for name,dep in spec['inputs'].items()}
            with np.errstate(over='ignore',invalid='ignore',under='ignore'):
                result=np.asarray(runtime().derive_field(spec,inputs),dtype=np.float64)
        if result.shape!=(stop-first,):raise ValidationError('storage prediction result shape mismatch')
        if not np.isfinite(result).all():raise PredictionOverflow('finite-input storage prediction overflow')
        memo[k]=result;return result
    return evaluate(key)

def create_values(group,rows,count,compression,*,resizable=False,info=None,key=None,width=None):
    step=max(1,min(count,ROWS)) if info is None else tile_rows(info,key,count,width)
    tiles=(count+step-1)//step
    values=group.create_dataset('values',shape=(rows,count),maxshape=(None,count) if resizable else None,dtype='uint64',chunks=(1,step),compression=compression,shuffle=True,fletcher32=True)
    values.attrs['codec']=CODEC
    values.attrs['role']=ROLE
    for name,dtype in (('prediction_modes','uint8'),('decoded_sha256','S64')):
        group.create_dataset(name,shape=(rows,tiles),maxshape=(None,tiles) if resizable else None,dtype=dtype,chunks=(1,max(1,min(tiles,4096))),fletcher32=True)
    return values

def validate_values(group,rows,count):
    values=group['values']
    if values.attrs.get('codec')!=CODEC or values.attrs.get('role')!=ROLE or values.dtype!=np.dtype('uint64') or values.shape!=(rows,count) or not values.chunks or values.chunks[0]!=1 or not 1<=values.chunks[1]<=ROWS:
        raise ValidationError('invalid integrity-coded stored values')
    tiles=(count+values.chunks[1]-1)//values.chunks[1]
    for name,dtype in (('prediction_modes','uint8'),('decoded_sha256','S64')):
        if name not in group or group[name].shape!=(rows,tiles) or group[name].dtype!=np.dtype(dtype):raise ValidationError('missing/invalid stored prediction integrity table')

def write_values(handle,group,info,key,actual,indices,coefficients,cases,groups,*,start_row=0,on_work=None,on_write=None,check=None):
    runtime()  # The new role explicitly requires storage grammar capability1.
    ds=group['values'];step=ds.chunks[1];width=coefficients.shape[1]+1
    for first in range(0,ds.shape[1],step):
        stop=min(first+step,ds.shape[1]);work=work_bytes(info,key,stop-first,width)
        if work>WORK_BYTES:raise WorkingSetError('storage prediction tile exceeds128MiB workspace')
        if on_work is not None:on_work(work)
        tables={}
        for primitive in primitives(info,key):
            if check is not None:check()
            tables[primitive]=admit_basis(handle['storage_prediction'][groups[primitive]]['responses'][first:stop],stop-first,width)
        source=lambda k,a,b:tables[k]
        for row,ci in enumerate(indices):
            if check is not None:check()
            weights=np.asarray(coefficients[int(ci)],dtype=np.float64)
            if not np.isfinite(weights).all():raise ValidationError('nonfinite storage prediction coefficients')
            source_values=np.asarray(actual[row,first:stop],dtype=np.float64)
            if not np.isfinite(source_values).all():raise ValidationError('nonfinite actual stored words')
            try:
                predicted=predict(info,key,first,stop,weights,cases[int(ci)].get('active_indices'),source,check,basis_verified=True)
                bits=predicted.view(np.uint64);mode=0
            except PredictionOverflow:
                bits=np.zeros(stop-first,dtype=np.uint64);mode=1
            ds[start_row+row,first:stop]=np.bitwise_xor(source_values.view(np.uint64),bits)
            group['prediction_modes'][start_row+row,first//step]=mode
            group['decoded_sha256'][start_row+row,first//step]=words_digest(source_values).encode('ascii')
            if on_write is not None:on_write((stop-first)*8)

def decode_tile(handle,group,info,key,row,case_index,first,stop,weights,active,groups,check=None,*,basis=None):
    runtime()
    if check is not None:check()
    step=group['values'].chunks[1];tile=first//step
    mode=int(group['prediction_modes'][row,tile])
    if mode==0:
        source=basis if basis is not None else lambda k,a,b:admit_basis(handle['storage_prediction'][groups[k]]['responses'][a:b],b-a,len(weights)+1)
        try:predicted=predict(info,key,first,stop,weights,active,source,check,basis_verified=True).view(np.uint64)
        except PredictionOverflow as error:raise ValidationError('storage prediction changed numeric branch; source words cannot be verified') from error
    elif mode==1:predicted=np.zeros(stop-first,dtype=np.uint64)
    else:raise ValidationError('unknown durable storage prediction mode')
    values=np.bitwise_xor(np.asarray(group['values'][row,first:stop],dtype=np.uint64),predicted).view(np.float64)
    expected=group['decoded_sha256'][row,tile]
    if not isinstance(expected,bytes) or len(expected)!=64 or words_digest(values).encode('ascii')!=expected or not np.isfinite(values).all():
        raise ValidationError('decoded stored source-word SHA256 mismatch (prediction environment or corrupt archive)')
    if check is not None:check()
    values.setflags(write=False);return values
