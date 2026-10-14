"""Provider queries. Physical recovery remains exclusively with the provider."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
import sys
from math import prod
import numpy as np
from .archive import Envelope
from .cache import ByteLRU
from .errors import (ResultsError, ValidationError, IdentityError, FailedCaseError,
    UnavailableFieldError, EmptySampleError, WorkingSetError, ClosedReaderError, UnknownCaseError,
    RecoveryError, RequestCancelled)
from .validation import SUCCESS

@dataclass(frozen=True)
class _Array:
    payload: bytes
    dtype: np.dtype
    shape: tuple
    @property
    def nbytes(self): return len(self.payload)
    def array(self): return np.frombuffer(self.payload,dtype=self.dtype).reshape(self.shape)

def _cancel(event):
    if event is not None:
        if not callable(getattr(event,'is_set',None)): raise TypeError('cancel_event must expose is_set()')
        if event.is_set(): raise RequestCancelled('result query cancelled')

def _size(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return sys.getsizeof(value)+value.nbytes*2
    if isinstance(value,Mapping):return sys.getsizeof(value)+sum(_size(k,seen)+_size(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)):return sys.getsizeof(value)+sum(_size(v,seen) for v in value)
    return sys.getsizeof(value)


def _transport_size(value,limit,path=None,seen=None):
    """Conservative expanded transport graph bound, including repeated aliases."""
    path=set() if path is None else path
    if id(value) in path:raise ValidationError('cyclic provider transport is unsupported')
    if seen is not None:
        if id(value) in seen:return 0
        seen.add(id(value))
    size=sys.getsizeof(value)
    if isinstance(value,np.ndarray):size+=value.nbytes*2
    elif isinstance(value,(Mapping,tuple,list)):
        path.add(id(value))
        try:
            children=(item for pair in value.items() for item in pair) if isinstance(value,Mapping) else iter(value)
            for child in children:
                size+=_transport_size(child,limit-size,path,seen)
                if size>limit:raise WorkingSetError('provider transport exceeds working cap')
        finally:path.remove(id(value))
    if size>limit:raise WorkingSetError('provider transport exceeds working cap')
    return size

def _snapshot(value,memo=None):
    memo={} if memo is None else memo
    if id(value) in memo:return memo[id(value)]
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValidationError('object arrays are not portable result metadata')
        result=_Array(value.tobytes(order='C'),value.dtype,value.shape)
    elif isinstance(value,Mapping):result={k:_snapshot(v,memo) for k,v in value.items()}
    elif isinstance(value,(tuple,list)):result=tuple(_snapshot(v,memo) for v in value)
    elif value is None or type(value) in (str,int,float,bool):result=value
    elif isinstance(value,np.generic):result=value.item()
    else:raise ValidationError('unsupported provider metadata type: '+type(value).__name__)
    memo[id(value)]=result
    return result

def _public(value):
    from types import MappingProxyType
    if isinstance(value,_Array):return value.array()
    if isinstance(value,dict):return MappingProxyType({k:_public(v) for k,v in value.items()})
    if isinstance(value,tuple):return tuple(_public(v) for v in value)
    return value

def _arrays(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,_Array):return value.nbytes
    if isinstance(value,dict):return sum(_arrays(v,seen) for v in value.values())
    if isinstance(value,tuple):return sum(_arrays(v,seen) for v in value)
    return 0

def reduce_samples(values,method='mean',*,sample_counts=None,nan_policy='propagate',
                   empty_policy='nan',vector_components=None,cancel_event=None,working_bytes=128<<20):
    """Reduce explicit trailing native sample axes, preserving signed max_abs ties."""
    if method not in ('mean','min','max','max_abs','magnitude'):raise ValueError('unknown sample reduction')
    if nan_policy not in ('propagate','omit','raise'):raise ValueError('unknown nan_policy')
    if empty_policy not in ('nan','zero','raise'):raise ValueError('unknown empty_policy')
    if isinstance(working_bytes,bool) or not isinstance(working_bytes,int):raise ValueError('working_bytes must be nonnegative integer')
    if working_bytes<0:raise WorkingSetError('sample reduction exceeds remaining working cap')
    if not isinstance(values,np.ndarray):raise ValidationError('native samples require ndarray')
    a=values
    if a.dtype.kind!='f' or a.ndim<1:raise ValidationError('native samples require a floating spatial-first array')
    width=prod(a.shape[1:]) if a.ndim>1 else 1
    if method=='magnitude':
        if tuple(vector_components or ())!=('ux','uy','uz') or a.shape!=(len(a),3):raise ValidationError('magnitude requires declared translation vector UX/UY/UZ with shape(N,3)')
    if a.nbytes*2+len(a)*24+4096>working_bytes:raise WorkingSetError('sample reduction exceeds working cap')
    counts=np.full(len(a),width,dtype=np.int64) if sample_counts is None else np.asarray(sample_counts)
    if counts.shape!=(len(a),) or counts.dtype.kind not in 'iu' or np.any(counts<0) or np.any(counts>width):raise ValidationError('sample_counts outside native row width')
    needed=a.nbytes*2+counts.nbytes+len(a)*a.dtype.itemsize*2+width*(a.dtype.itemsize+1)+4096
    if needed>working_bytes:raise WorkingSetError('sample reduction exceeds working cap')
    if method=='magnitude':
        if np.any(counts!=3):raise ValidationError('translation magnitude requires all three vector components')
        if np.isnan(a).any() and nan_policy in ('raise','omit'):raise ValidationError('translation magnitude has missing vector components')
        _cancel(cancel_event);out=np.linalg.norm(a,axis=1);_cancel(cancel_event)
        return np.frombuffer(out.tobytes(),dtype=out.dtype).reshape(out.shape)
    rows=a.reshape(len(a),width);out=np.empty(len(a),dtype=a.dtype)
    for i,n in enumerate(counts):
        _cancel(cancel_event)
        row=rows[i,:int(n)]
        missing=np.isnan(row)
        if missing.any():
            if nan_policy=='raise':raise ValidationError('NaN in actual native samples')
            if nan_policy=='propagate':out[i]=np.nan;continue
            row=row[~missing]
            if not len(row):
                # Counted all-NaN samples are missing numerical evidence,
                # distinct from structurally empty native rows.
                out[i]=np.nan;continue
        if not len(row):
            if empty_policy=='raise':raise EmptySampleError('native sample row is empty')
            out[i]=0 if empty_policy=='zero' else np.nan;continue
        if method=='max_abs':out[i]=row[int(np.argmax(np.abs(row)))]
        elif method=='mean':out[i]=np.mean(row)
        elif method=='min':out[i]=np.min(row)
        else:out[i]=np.max(row)
    _cancel(cancel_event)
    return np.frombuffer(out.tobytes(),dtype=out.dtype).reshape(out.shape)

def vector_magnitude(values,*,axis=-1,components,working_bytes=128<<20,cancel_event=None):
    """Rendering-only norm of an explicitly declared stored vector axis."""
    if not isinstance(values,np.ndarray) or values.dtype.kind!='f' or values.ndim<1:raise ValidationError('vector magnitude requires floating ndarray')
    if isinstance(axis,bool) or not isinstance(axis,int) or not -values.ndim<=axis<values.ndim:raise ValidationError('invalid vector axis')
    if not isinstance(components,tuple) or any(not isinstance(c,str) or not c for c in components) or len(components)!=values.shape[axis]:raise ValidationError('declared vector components disagree with axis width')
    if isinstance(working_bytes,bool) or not isinstance(working_bytes,int) or working_bytes<0:raise ValueError('working_bytes must be nonnegative integer')
    if values.nbytes*4+4096>working_bytes:raise WorkingSetError('vector magnitude exceeds working cap')
    _cancel(cancel_event);result=np.linalg.norm(values,axis=axis);_cancel(cancel_event)
    return _Array(result.tobytes(order='C'),result.dtype,result.shape).array()

def absolute_sample_max(values,*,sample_counts=None,nan_policy='omit',empty_policy='nan',working_bytes=128<<20,cancel_event=None):
    """Explicit generic absolute native-sample maximum for rendering."""
    if not isinstance(values,np.ndarray):raise ValidationError('sample maximum requires ndarray')
    if values.nbytes*4+4096>working_bytes:raise WorkingSetError('absolute sample maximum exceeds working cap')
    reduced=reduce_samples(values,'max_abs',sample_counts=sample_counts,nan_policy=nan_policy,empty_policy=empty_policy,working_bytes=working_bytes,cancel_event=cancel_event)
    _cancel(cancel_event);result=np.abs(reduced);_cancel(cancel_event)
    return _Array(result.tobytes(order='C'),result.dtype,result.shape).array()

class _Descriptor(dict):
    """Private normalized descriptor with a once-validated layout."""
    __slots__ = ('layout',)

@dataclass(frozen=True)
class EnvelopeSelection:
    values: np.ndarray
    case_ids: np.ndarray
    admitted_case_ids: tuple=()
    failed_case_ids: tuple=()
    requested_case_ids: tuple=()

class MemoryResultsReader:
    """A model-bound snapshot of metadata; provider owns numerical recovery/lifetime."""
    def __init__(self,provider,*,cache_bytes=2<<30,working_bytes=128<<20,expected_model_identity=None):
        if isinstance(working_bytes,bool) or not isinstance(working_bytes,int) or working_bytes<0:raise ValueError('working_bytes must be nonnegative integer')
        self.working_bytes=working_bytes;self._cache=ByteLRU(cache_bytes);self._lock=RLock();self._closed=False
        self._working_peak=0;self._reads=0;self._rows=0;self._reserved_bytes=0;self._provider=provider
        if not callable(getattr(provider,'read_field',None)):raise ValidationError('provider requires read_field()')
        method=getattr(provider,'metadata',None)
        meta=method() if callable(method) else getattr(provider,'metadata',{})
        if not isinstance(meta,Mapping):raise ValidationError('provider metadata must be a mapping')
        meta=dict(meta)
        for name,attribute in (('descriptors','components'),('cases','cases')):
            if name not in meta:
                value=getattr(provider,attribute,None);meta[name]=value() if callable(value) else value
        identity=meta.get('model_identity',getattr(provider,'model_identity',None))
        if not isinstance(identity,str) or not identity:raise ValidationError('provider requires model_identity')
        if expected_model_identity is not None and identity!=expected_model_identity:raise IdentityError('provider model identity mismatch')
        meta['model_identity']=identity
        if not isinstance(meta.get('cases'),(tuple,list)) or not isinstance(meta.get('descriptors'),(tuple,list)):raise ValidationError('provider requires cases and descriptors')
        self._normalization_input_bytes=_transport_size(meta,max(0,(self.working_bytes-4096)//2),seen=set());self._normalization_vectors={}
        self._reserve(self._normalization_input_bytes*2+4096);self._meta=_snapshot(meta)
        self._metadata_cache_owner=object()
        self._metadata_size=_size(self._meta);self._metadata_arrays=_arrays(self._meta);self._normalization_added_bytes=0
        self._cases={};self._pairs={}
        for case in self._meta['cases']:
            if not isinstance(case,dict):raise ValidationError('provider case must be a mapping')
            cid=case.get('case_id',case.get('id'))
            if not isinstance(cid,str) or not cid or cid in self._cases or not isinstance(case.get('status'),str):raise ValidationError('invalid/duplicate provider case identity or status')
            case['case_id']=cid;case.setdefault('id',cid);self._cases[cid]=case
        expanded=[];descriptor_keys=set()
        for d in self._meta['descriptors']:
            if not isinstance(d,dict):raise ValidationError('provider descriptor must be a mapping')
            d=self._normalise_descriptor(d)
            components=(d['component'],) if 'component' in d else d.get('components',())
            if not isinstance(components,tuple):raise ValidationError('descriptor components must be a sequence')
            for component in components:
                entry=_Descriptor(d,component=component);entry.layout=d.layout;pair=(d.get('quantity_id'),component)
                if any(not isinstance(x,str) or not x for x in pair) or pair in self._pairs or not isinstance(d.get('location'),str):raise ValidationError('invalid/duplicate descriptor')
                entry.setdefault('key',pair[0]+':'+pair[1])
                if not isinstance(entry['key'],str) or not entry['key'] or entry['key'] in descriptor_keys:raise ValidationError('invalid/duplicate descriptor key')
                descriptor_keys.add(entry['key']);self._pairs[pair]=entry;expanded.append(entry)
                self._layout(entry)
        for case in self._cases.values():
            unavailable=case.get('unavailable_components',())
            if not isinstance(unavailable,tuple) or any(not isinstance(key,str) or key not in descriptor_keys for key in unavailable) or len(set(unavailable))!=len(unavailable):raise ValidationError('invalid/unknown/duplicate unavailable_components')
        self._meta['descriptors']=tuple(expanded)
        self._case_layouts={}
        bases={d['key']:d for d in expanded}
        case_overrides=self._meta.get('case_descriptors',{})
        if not isinstance(case_overrides,dict):raise ValidationError('case_descriptors must be a mapping')
        for cid,overrides in case_overrides.items():
            if cid not in self._cases:raise ValidationError('unknown case in case_descriptors')
            if not isinstance(overrides,dict):raise ValidationError('case descriptor entries must be mappings')
            for key,override in overrides.items():
                if not isinstance(override,dict):raise ValidationError('case descriptor override must be a mapping')
                if key not in bases:raise ValidationError('unknown component key in case_descriptors')
                if key in bases:
                    merged=self._normalise_descriptor(dict(bases[key],**override))
                    self._require_invariants(bases[key],merged)
                    overrides[key]={name:merged[name] for name in override}
                    self._case_layouts[(cid,key)]=merged
        self._normalization_input_bytes=0;self._normalization_vectors=None;self._normalization_added_bytes=0
        self._metadata_size=_size(self._meta);self._metadata_arrays=_arrays(self._meta)
        self._cache.set_external_bytes(self._metadata_cache_owner,self._metadata_arrays)
    def _ensure(self):
        if self._closed:raise ClosedReaderError('result reader is closed')
    def _reserve(self,n):
        n+=self._reserved_bytes
        if n>self.working_bytes:raise WorkingSetError(f'request needs at least {n} bytes; working cap is {self.working_bytes}')
        self._working_peak=max(self._working_peak,int(n))
    def __enter__(self):self._ensure();return self
    def __exit__(self,*args):self.close()
    def close(self):
        with self._lock:
            self._closed=True;self._cache.clear();self._cache._external={};self._provider=None;self._meta={};self._cases={};self._pairs={};self._case_layouts={}
    def clear_cache(self):
        with self._lock:self._ensure();self._cache.clear()
    def set_external_cache_bytes(self,owner_key,byte_count):
        with self._lock:
            self._ensure()
            if owner_key is self._metadata_cache_owner:raise ValidationError('metadata reservation is private')
            self._cache.set_external_bytes(('public',owner_key),byte_count)
    def resize_cache(self,cache_bytes):
        """Resize this reader only; live external reservations cannot be evicted."""
        if isinstance(cache_bytes,bool) or not isinstance(cache_bytes,int) or cache_bytes<0:raise ValueError('cache_bytes must be a nonnegative integer')
        with self._lock:
            self._ensure()
            with self._cache._lock:
                external=sum(getattr(self._cache,'_external',{}).values())
                if external>cache_bytes:raise WorkingSetError('external reservations exceed proposed cache ceiling')
                self._cache.capacity=cache_bytes
                while self._cache.resident_bytes+external>cache_bytes and self._cache._values:
                    _,(_,removed)=self._cache._values.popitem(last=False)
                    self._cache.resident_bytes-=removed
    @property
    def diagnostics(self):
        with self._lock:
            self._ensure();external=sum(getattr(self._cache,'_external',{}).values())
            return dict(cache_bytes=self._cache.capacity,working_bytes=self.working_bytes,cache_resident_bytes=self._cache.resident_bytes+external,cache_peak_bytes=self._cache.peak_bytes,external_cache_bytes=external,working_peak_bytes=self._working_peak,cache_hits=self._cache.hits,cache_misses=self._cache.misses,reads=self._reads,provider_rows=self._rows)
    def metadata(self):
        with self._lock:self._ensure();return _public(self._meta)
    metadata_summary=metadata
    def cases(self):
        with self._lock:self._ensure();return _public(tuple(self._cases.values()))
    case_summaries=cases
    def case_metadata(self,case_id):
        with self._lock:self._ensure();return _public(self._case(case_id,admit=False))
    def components(self,quantity_id=None):
        with self._lock:self._ensure();return _public(tuple(d for (q,c),d in self._pairs.items() if quantity_id is None or q==quantity_id))
    def _case(self,cid,admit=True):
        try:c=self._cases[cid]
        except (KeyError,TypeError):raise UnknownCaseError(cid) from None
        if admit and c['status'] not in SUCCESS:raise FailedCaseError(f'case {cid!r} did not complete',failed_case_ids=(cid,),requested_case_ids=(cid,))
        return c
    def _descriptor(self,q,c):
        try:return self._pairs[(q,c)]
        except (KeyError,TypeError):raise UnavailableFieldError(f'unknown quantity/component {q!r}/{c!r}') from None
    def _normalise_descriptor(self,d,*,workspace=0):
        # Admit complete vector conversion/validation once, not just selected
        # rows. Immutable normalized buffers are reused by every static read.
        scratch=4096
        for name in ('ids','sample_counts','present'):
            value=d.get(name)
            if isinstance(value,_Array):scratch+=value.nbytes*2
            elif isinstance(value,tuple):scratch+=len(value)*32
        memo=self._normalization_vectors
        self._reserve(self._metadata_size+self._metadata_arrays+_arrays(d)+scratch+workspace+self._normalization_input_bytes+self._normalization_added_bytes)
        result=_Descriptor(d)
        for name in ('ids','sample_counts','present'):
            value=result.get(name)
            if isinstance(value,tuple):
                if name=='ids' and any(type(i) is not int for i in value):raise ValidationError('descriptor ids require integers')
                cached=None if memo is None else memo.get((name,id(value)))
                if cached is not None and cached[0] is value:
                    result[name]=cached[1];continue
                a=np.asarray(value)
                if not len(value):a=np.empty(0,dtype=bool if name=='present' else np.int64)
                if a.ndim!=1 or a.dtype.kind not in ('b' if name=='present' else 'iu'):raise ValidationError('invalid descriptor '+name)
                result[name]=_Array(a.tobytes(order='C'),a.dtype,a.shape)
                if memo is not None:
                    memo[(name,id(value))]=(value,result[name]);self._normalization_added_bytes+=result[name].nbytes+sys.getsizeof(result[name])+sys.getsizeof(result[name].payload)
        result.layout=self._validate_layout(result)
        if memo is not None:self._normalization_added_bytes+=sys.getsizeof(result)
        return result
    def _layout(self,d):
        if isinstance(d,_Descriptor):return d.layout
        return self._validate_layout(d)
    def _validate_layout(self,d):
        ids=d.get('ids')
        count=d.get('spatial_count')
        if ids is not None and not isinstance(ids,(_Array,tuple)):raise ValidationError('descriptor ids require frozen integer vector')
        if isinstance(ids,_Array):
            a=ids.array()
            if a.ndim!=1 or a.dtype.kind not in 'iu':raise ValidationError('descriptor ids require integer spatial axis')
            count=len(a)
        elif isinstance(ids,tuple):
            if any(type(i) is not int for i in ids):raise ValidationError('descriptor ids require integers')
            count=len(ids)
        if isinstance(count,bool) or not isinstance(count,int) or count<0:raise ValidationError('descriptor requires ids or spatial_count')
        if 'spatial_count' in d:
            declared=d['spatial_count']
            if isinstance(declared,bool) or not isinstance(declared,int) or declared<0 or declared!=count:raise ValidationError('spatial_count contradicts declared entity IDs')
        counts=d.get('sample_counts');counts=counts.array() if isinstance(counts,_Array) else counts
        aliases=[]
        for name in ('value_shape','sample_shape'):
            value=d.get(name)
            if value is not None:
                if not isinstance(value,(tuple,list)) or any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in value):raise ValidationError('invalid native sample shape')
                aliases.append(tuple(value))
        if len(aliases)==2 and aliases[0]!=aliases[1]:raise ValidationError('conflicting value_shape/sample_shape aliases')
        shape=aliases[0] if aliases else None
        if shape is None:
            if counts is not None:
                ca=np.asarray(counts)
                if ca.shape!=(count,) or ca.dtype.kind not in 'iu' or np.any(ca<0):raise ValidationError('invalid sample_counts')
                shape=(int(ca.max(initial=0)),)
            else:shape=()
        if not isinstance(shape,(tuple,list)):raise ValidationError('native sample shape must be a sequence')
        shape=tuple(shape)
        if any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in shape):raise ValidationError('invalid native sample shape')
        try:dtype=np.dtype(d.get('dtype','float64'))
        except (TypeError,ValueError) as error:raise ValidationError('invalid provider dtype') from error
        if dtype.kind!='f':raise ValidationError('provider field dtype must be floating')
        if counts is not None:
            ca=np.asarray(counts)
            if ca.shape!=(count,) or ca.dtype.kind not in 'iu' or np.any(ca<0) or np.any(ca>(prod(shape) if shape else 1)):raise ValidationError('sample_counts outside declared native shape')
        present=d.get('present')
        if present is not None:
            pa=present.array() if isinstance(present,_Array) else np.asarray(present)
            if pa.shape!=(count,) or pa.dtype.kind!='b':raise ValidationError('present mask requires bool spatial axis')
        return count,shape,dtype,counts
    def _require_invariants(self,base,current):
        bc,bs,bt,bn=self._layout(base);cc,cs,ct,cn=self._layout(current)
        if bc!=cc or bt!=ct:raise ValidationError('case descriptor changed spatial axis or precision')
        if current.get('spatial_count',cc)!=base.get('spatial_count',bc):raise ValidationError('case descriptor changed spatial_count')
        for name in ('quantity_id','component','components','key','location','unit','basis','vector_components','sampling'):
            if current.get(name)!=base.get(name):raise ValidationError('case descriptor changed '+name)
        left=base.get('ids');right=current.get('ids')
        if left is right:return
        if left is None or right is None:raise ValidationError('case descriptor changed entity IDs')
        left=left.array() if isinstance(left,_Array) else left
        right=right.array() if isinstance(right,_Array) else right
        if not np.array_equal(left,right):raise ValidationError('case descriptor changed entity IDs')
    def _select(self,count,selection):
        if selection is None:return slice(0,count,1),count,('slice',0,count,1),0
        if isinstance(selection,slice):
            try:start,stop,step=selection.indices(count)
            except (ValueError,TypeError) as error:raise ValidationError('invalid spatial slice') from error
            n=len(range(start,stop,step));return slice(start,None if step<0 and stop==-1 else stop,step),n,('slice',start,stop,step),0
        if isinstance(selection,np.ndarray):
            if selection.ndim!=1 or selection.dtype.kind not in 'iu':raise ValidationError('selection requires one integer spatial axis')
            n=len(selection);selector_bytes=n*18+selection.nbytes
            self._reserve(selector_bytes+4096);a=selection
        elif isinstance(selection,(tuple,list)):
            n=len(selection);selector_bytes=n*26
            self._reserve(selector_bytes+4096)
            if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in selection):raise ValidationError('selection requires integers')
            try:a=np.asarray(selection,dtype=np.int64)
            except (OverflowError,ValueError,TypeError) as error:raise ValidationError('selection outside integer domain') from error
        else:raise ValidationError('selection requires an integer array or sequence')
        if np.any(a<0) or np.any(a>=count):raise ValidationError('selection outside spatial axis')
        a=np.array(a,dtype=np.int64,copy=True);a.flags.writeable=False
        return a,n,('indices',a.tobytes()),selector_bytes
    def _field_layout(self,cid,q,c,base,*,stored_only=False):
        descriptor=self._case_layouts.get((cid,base['key']),base)
        normalization_work=0
        if descriptor.get('availability')=='unavailable':raise UnavailableFieldError('case field is unavailable',unavailable_case_ids=(cid,))
        if stored_only and descriptor.get('availability')!='stored':raise UnavailableFieldError('stored-only provider query requires explicit stored availability',unavailable_case_ids=(cid,))
        method=getattr(self._provider,'field_descriptor',None)
        if callable(method):
            bound=descriptor.get('provider_work_bytes',getattr(self._provider,'provider_work_bytes',None))
            if isinstance(bound,bool) or not isinstance(bound,int) or bound<0:raise ValidationError('field_descriptor requires declared provider_work_bytes')
            self._reserve(bound+4096)
            try:extra=method(cid,q,c)
            except ResultsError:raise
            except KeyError as error:raise UnavailableFieldError('provider field descriptor unavailable',unavailable_case_ids=(cid,)) from error
            except (ValueError,OSError) as error:raise ValidationError('provider field_descriptor failed') from error
            if not isinstance(extra,Mapping):raise ValidationError('field_descriptor must return mapping')
            normalization_work=_transport_size(extra,max(0,(self.working_bytes-self._reserved_bytes-bound-4096)//2),seen=set())*2+bound+4096
            self._reserve(normalization_work);descriptor=dict(descriptor,**_snapshot(extra))
        descriptor=descriptor if isinstance(descriptor,_Descriptor) else self._normalise_descriptor(descriptor,workspace=normalization_work)
        self._require_invariants(base,descriptor)
        if descriptor.get('availability')=='unavailable':raise UnavailableFieldError('case field is unavailable',unavailable_case_ids=(cid,))
        return descriptor
    @staticmethod
    def _validate_policies(reduction,nan_policy,empty_policy):
        if reduction not in (None,'mean','min','max','max_abs','magnitude'):raise ValueError('unknown sample reduction')
        if nan_policy not in ('propagate','omit','raise'):raise ValueError('unknown nan_policy')
        if empty_policy not in ('nan','zero','raise'):raise ValueError('unknown empty_policy')
    def _read_field(self,case_id,quantity_id,component,selection=None,*,reduction=None,nan_policy='propagate',empty_policy='nan',cancel_event=None,_return_presence=False,_stored_only=False):
        with self._lock:
            self._ensure();_cancel(cancel_event);self._validate_policies(reduction,nan_policy,empty_policy);self._case(case_id,admit=not _stored_only)
            base=self._descriptor(quantity_id,component)
            if base['key'] in self._cases[case_id].get('unavailable_components',()):raise UnavailableFieldError('case field is unavailable',unavailable_case_ids=(case_id,))
            d=self._field_layout(case_id,quantity_id,component,base,stored_only=_stored_only)
            _cancel(cancel_event)
            if _stored_only and d.get('availability')!='stored':raise UnavailableFieldError('provider changed stored-only availability',unavailable_case_ids=(case_id,))
            count,shape,dtype,counts=self._layout(d)
            if reduction=='magnitude' and (tuple(d.get('vector_components',()))!=('ux','uy','uz') or shape!=(3,)):raise ValidationError('magnitude requires declared translation vector')
            sel,n,identity,selector_bytes=self._select(count,selection)
            native_width=prod(shape) if shape else 1
            raw_bytes=n*native_width*dtype.itemsize
            sample_presence=_return_presence and reduction is None and counts is not None
            # Explicit scratch allowance for the native-slot mask, ordinal
            # vector and selected counts; padding is not source NaN evidence.
            mask_bytes=(n*native_width+native_width*8+n*8) if sample_presence and n else 0
            reserve=raw_bytes*4+selector_bytes+4096+n*32+_arrays(d)+mask_bytes
            provider_work=d.get('provider_work_bytes',getattr(self._provider,'provider_work_bytes',None))
            if isinstance(provider_work,bool) or not isinstance(provider_work,int) or provider_work<0:raise ValidationError('provider_work_bytes must be nonnegative integer')
            self._reserve(reserve+provider_work)
            presence=d.get('present')
            presence=presence.array() if isinstance(presence,_Array) else presence
            presence=np.ones(n,dtype=bool) if presence is None else np.asarray(presence)[sel]
            envelope_presence=presence
            if sample_presence:
                if n:
                    envelope_presence=np.arange(native_width,dtype=np.int64)[None,:]<np.asarray(counts)[sel,None]
                    envelope_presence &= presence[:,None]
                    envelope_presence=envelope_presence.reshape((n,)+shape)
                else:envelope_presence=np.empty((n,)+shape,dtype=bool)
                _cancel(cancel_event)
            if _return_presence and reduction is not None and empty_policy=='nan':
                # A zero-source-sample result is missing evidence, not a source
                # NaN. Keep explicit zero/raise policies and actual NaNs intact.
                envelope_presence=presence & ((np.asarray(counts)[sel]>0) if counts is not None else native_width>0)
            def result(values):
                return (values,envelope_presence) if _return_presence else values
            key=(self._meta['model_identity'],self._meta.get('mesh_identity'),case_id,quantity_id,component,identity,reduction,nan_policy,empty_policy,_stored_only)
            cached=self._cache.get(key)
            if cached is not None:_cancel(cancel_event);return result(cached.array())
            try:value=self._provider.read_field(case_id,quantity_id,component,selection=sel,cancel_event=cancel_event)
            except ResultsError:raise
            except KeyError as error:raise UnavailableFieldError(str(error),unavailable_case_ids=(case_id,)) from error
            except (ValueError,OSError) as error:raise ValidationError('provider field query failed') from error
            _cancel(cancel_event)
            if not isinstance(value,np.ndarray) or value.shape!=(n,)+shape or value.dtype!=dtype:raise ValidationError('provider field shape/dtype differs from declared layout')
            self._reads+=1;self._rows+=n
            if not _stored_only or (reduction is None and nan_policy=='raise'):
                flat=value.reshape(n,native_width);selected_counts=None if counts is None else np.asarray(counts)[sel]
                for i in range(n):
                    _cancel(cancel_event)
                    if not presence[i]:continue
                    actual=flat[i,:native_width if selected_counts is None else int(selected_counts[i])]
                    if not _stored_only and np.isinf(actual).any():raise ValidationError('infinite provider engineering values')
                    if reduction is None and nan_policy=='raise' and np.isnan(actual).any():raise ValidationError('NaN in actual native samples')
            if reduction is not None:
                sliced=None if counts is None else np.asarray(counts)[sel]
                reduction_budget=self.working_bytes-selector_bytes-provider_work-self._reserved_bytes-_arrays(d)
                if reduction_budget<0:raise WorkingSetError('sample reduction exceeds remaining working cap')
                if presence.all():
                    value=reduce_samples(value,reduction,sample_counts=sliced,vector_components=d.get('vector_components'),nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event,working_bytes=reduction_budget)
                else:
                    if reduction_budget<n*dtype.itemsize+raw_bytes:raise WorkingSetError('sample reduction exceeds remaining working cap')
                    reduced=np.full(n,np.nan,dtype=value.dtype)
                    reduced[presence]=reduce_samples(value[presence],reduction,sample_counts=None if sliced is None else sliced[presence],vector_components=d.get('vector_components'),nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event,working_bytes=reduction_budget-reduced.nbytes-raw_bytes)
                    value=reduced
            if not presence.all():
                value=np.array(value,copy=True);value[~presence]=np.nan
            frozen=_Array(value.tobytes(order='C'),value.dtype,value.shape)
            self._cache.put(key,frozen,key_bytes=len(identity[1]) if identity[0]=='indices' else 0);_cancel(cancel_event);return result(frozen.array())
    def read_field(self,case_id,quantity_id,component,selection=None,*,reduction=None,nan_policy='propagate',empty_policy='nan',cancel_event=None):
        return self._read_field(case_id,quantity_id,component,selection,reduction=reduction,nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event)
    def read_stored_field(self,case_id,quantity_id,component,selection=None,*,reduction=None,nan_policy='propagate',empty_policy='nan',cancel_event=None):
        """Actual stored snapshot only; source status/certification stay unchanged."""
        return self._read_field(case_id,quantity_id,component,selection,reduction=reduction,nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event,_stored_only=True)
    def _envelope(self,quantity_id,component,selection,case_ids,*,mode,tie_policy,strict_failed,reduction,nan_policy,empty_policy,cancel_event):
        with self._lock:
            self._ensure();_cancel(cancel_event);d=self._descriptor(quantity_id,component)
            if isinstance(case_ids,(str,bytes)):raise ValidationError('case_ids requires a sequence of case identities')
            try:chosen=tuple(self._cases) if case_ids is None else tuple(case_ids)
            except (TypeError,ValueError) as error:raise ValidationError('case_ids requires a sequence of identities') from error
            if not chosen or any(not isinstance(cid,str) or not cid for cid in chosen):raise ValidationError('envelope requires nonempty string case identities')
            self._validate_policies(reduction,nan_policy,empty_policy)
            if len(set(chosen))!=len(chosen):raise ValidationError('envelope case IDs must be unique')
            if nan_policy not in ('propagate','omit','raise'):raise ValueError('unknown nan_policy')
            if type(strict_failed) is not bool:raise TypeError('strict_failed must be a boolean')
            if tie_policy is None:tie_policy=d.get('tie_policy','first')
            if tie_policy not in ('first','last','exact_earliest_input_index'):raise ValueError('unknown tie_policy')
            admitted=[];failed=[]
            for cid in chosen:
                c=self._case(cid,admit=False)
                (admitted if c['status'] in SUCCESS else failed).append(cid)
            if strict_failed and failed:raise FailedCaseError('failed cases requested: '+repr(tuple(failed)),failed_case_ids=failed,requested_case_ids=chosen)
            if not admitted:raise FailedCaseError('no completed cases requested',failed_case_ids=failed,requested_case_ids=chosen)
            if tie_policy=='exact_earliest_input_index':
                indices=[self._cases[cid].get('input_index') for cid in self._cases]
                if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):raise ValidationError('authoritative input_index must be unique nonnegative integer')
                admitted.sort(key=lambda cid:self._cases[cid]['input_index'])
            count,shape,dtype,counts=self._layout(d)
            if reduction=='magnitude' and (tuple(d.get('vector_components',()))!=('ux','uy','uz') or shape!=(3,)):raise ValidationError('magnitude requires declared translation vector')
            sel,n,identity,selector_bytes=self._select(count,selection)
            output_shape=(n,) if reduction is not None else (n,)+shape
            items=prod(output_shape);width=max(1,max(len(cid) for cid in chosen))
            state_bytes=items*(dtype.itemsize*2+8*width+8)+selector_bytes+4096
            self._reserve(state_bytes)
            previous=self._reserved_bytes;self._reserved_bytes+=state_bytes
            low=high=low_ids=high_ids=None;seen=np.zeros(output_shape,dtype=bool);missing=[]
            try:
                for cid in admitted:
                    _cancel(cancel_event)
                    try:v,present=self._read_field(cid,quantity_id,component,sel,reduction=reduction,nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event,_return_presence=True)
                    except EmptySampleError:raise
                    except UnavailableFieldError:missing.append(cid);continue
                    if v.shape!=output_shape or v.dtype!=dtype:raise ValidationError('envelope cases have incompatible native sample layouts or precision')
                    active=present if present.shape==v.shape else np.broadcast_to(present.reshape((n,)+(1,)*(v.ndim-1)),v.shape)
                    if low is None:
                        low=np.full(v.shape,np.nan,dtype=v.dtype);high=low.copy()
                        low_ids=np.full(v.shape,'',dtype=f'U{width}');high_ids=low_ids.copy()
                    nan=active & np.isnan(v)
                    if nan_policy=='raise' and nan.any():raise ValidationError('NaN in actual envelope values')
                    if nan_policy=='omit':active=active & ~nan
                    fresh=active & ~seen
                    if mode=='max_abs':
                        lt=active & seen & (np.abs(v)>=np.abs(low) if tie_policy=='last' else np.abs(v)>np.abs(low))
                        selected=fresh | lt;low[selected]=v[selected];low_ids[selected]=cid
                    else:
                        lt=active & seen & (v<=low if tie_policy=='last' else v<low)
                        gt=active & seen & (v>=high if tie_policy=='last' else v>high)
                        low[fresh | lt]=v[fresh | lt];low_ids[fresh | lt]=cid
                        high[fresh | gt]=v[fresh | gt];high_ids[fresh | gt]=cid
                    if nan_policy=='propagate':
                        low[nan]=np.nan;high[nan]=np.nan
                    seen|=active
                    low_ids[np.isnan(low)]='';high_ids[np.isnan(high)]=''
                if missing:raise UnavailableFieldError('completed cases lack requested component',unavailable_case_ids=tuple(missing))
                _cancel(cancel_event)
                def frozen(a):return _Array(a.tobytes(order='C'),a.dtype,a.shape).array()
                coverage=dict(admitted_case_ids=tuple(admitted),failed_case_ids=tuple(failed),requested_case_ids=chosen)
                # Snapshot conversion lives simultaneously with four outputs.
                self._reserve(state_bytes)
                if mode is None:return Envelope(*(frozen(a) for a in (low,high,low_ids,high_ids)),**coverage)
                values,ids=(high,high_ids) if mode=='max' else (low,low_ids)
                return EnvelopeSelection(frozen(values),frozen(ids),**coverage)
            finally:self._reserved_bytes=previous
    def envelope(self,quantity_id,component,selection=None,case_ids=None,*,tie_policy=None,strict_failed=False,reduction=None,nan_policy='propagate',empty_policy='nan',cancel_event=None):
        return self._envelope(quantity_id,component,selection,case_ids,mode=None,tie_policy=tie_policy,strict_failed=strict_failed,reduction=reduction,nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event)
    def read_envelope(self,quantity_id,component,selection=None,case_ids=None,*,mode='max',tie_policy=None,strict_failed=False,reduction=None,nan_policy='propagate',empty_policy='nan',cancel_event=None):
        if mode not in ('min','max','max_abs'):raise ValueError('unknown envelope mode')
        return self._envelope(quantity_id,component,selection,case_ids,mode=mode,tie_policy=tie_policy,strict_failed=strict_failed,reduction=reduction,nan_policy=nan_policy,empty_policy=empty_policy,cancel_event=cancel_event)

    def _passthrough(self,name,*args,cancel_event=None,**kwargs):
        with self._lock:
            self._ensure();_cancel(cancel_event);method=getattr(self._provider,name,None)
            if not callable(method):raise UnavailableFieldError('provider does not expose '+name)
            workspace=getattr(self._provider,'provider_work_bytes',self._meta.get('provider_work_bytes'))
            if isinstance(workspace,bool) or not isinstance(workspace,int) or workspace<0:raise ValidationError('pass-through requires declared provider_work_bytes')
            # The owner declaration includes every new provider allocation,
            # including sizing/preflight. Zero denotes pre-retained results.
            self._reserve(workspace+4096)
            preflight=getattr(self._provider,'passthrough_work_bytes',None)
            if preflight is not None:
                if not callable(preflight):raise ValidationError('passthrough_work_bytes must be callable')
                try:workspace=preflight(name,*args,**kwargs)
                except ResultsError:raise
                except (ValueError,OSError,KeyError) as error:raise ValidationError('provider transport preflight failed') from error
                _cancel(cancel_event)
                if isinstance(workspace,bool) or not isinstance(workspace,int) or workspace<0:raise ValidationError('invalid pass-through workspace declaration')
                self._reserve(workspace+4096)
            try:value=method(*args,**kwargs)
            except ResultsError:raise
            except (ValueError,OSError,KeyError) as error:raise ValidationError('provider '+name+' query failed') from error
            _cancel(cancel_event)
            remaining=self.working_bytes-self._reserved_bytes-workspace-4096
            if remaining<0:raise WorkingSetError('provider transport exceeds remaining working cap')
            size=_transport_size(value,remaining//3)
            self._reserve(workspace+size*3+4096)
            snapshot=_snapshot(value);_cancel(cancel_event)
            result=_public(snapshot);_cancel(cancel_event)
            return result
    def geometry(self,selection=None,*,cancel_event=None):return self._passthrough('geometry',selection=selection,cancel_event=cancel_event)
    def associations(self,quantity_id,component,selection=None,*,cancel_event=None):
        method=getattr(self._provider,'associations',None)
        if callable(method):return self._passthrough('associations',quantity_id,component,selection=selection,cancel_event=cancel_event)
        with self._lock:
            self._ensure();_cancel(cancel_event);d=self._descriptor(quantity_id,component);count,shape,dtype,counts=self._layout(d)
            ids=d.get('ids')
            if ids is None:raise UnavailableFieldError('descriptor has no persistent IDs')
            ids=ids.array() if isinstance(ids,_Array) else np.asarray(ids,dtype=np.int64)
            sel,n,key,budget=self._select(count,selection);self._reserve(n*16+budget+4096)
            value=ids[sel];return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    def table(self,*args,cancel_event=None,**kwargs):return self._passthrough('table',*args,cancel_event=cancel_event,**kwargs)
    def table_keys(self):return self._passthrough('table_keys')
    def frames(self):return self._passthrough('frames')
    def read_history(self,*args,cancel_event=None,**kwargs):return self._passthrough('read_history',*args,cancel_event=cancel_event,**kwargs)
    def query(self,*args,cancel_event=None,**kwargs):return self._passthrough('query',*args,cancel_event=cancel_event,**kwargs)


def open_memory_results(provider,*,cache_bytes=2<<30,working_bytes=128<<20,expected_model_identity=None):
    return MemoryResultsReader(provider,cache_bytes=cache_bytes,working_bytes=working_bytes,expected_model_identity=expected_model_identity)
