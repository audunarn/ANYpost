"""Read-only ANYfem v1 sidecars, with explicit frame and sample semantics."""
import json
from copy import deepcopy
import numpy as np
from .archive import ResultsReader, _text, _freeze, _io_boundary, _cancellable
from .errors import ValidationError, UnavailableFieldError, UnknownCaseError, FailedCaseError

class LegacyResultsReader(ResultsReader):
    def _load_metadata(self):
        h=self._handle
        if _text(h.attrs.get("schema",""))!="anyfem.result" or h.attrs.get("version")!=1 or not h.attrs.get("complete",False):
            raise ValidationError("unsupported or incomplete ANYfem result sidecar")
        self.model_identity=_text(h.attrs.get("model_hash",""))
        self.mesh_identity=_text(h.attrs.get("mesh_hash",""))
        self._legacy_fields={}; self._legacy_frame_maps={}; self._legacy_scoped_cases={}; self._counts={}; self._descriptors={}; self._pairs={}
        frame_ds=h.get("frames/values")
        self._frames=() if frame_ds is None else tuple(float(v) for v in frame_ds[()])
        maximum=max(1,len(self._frames))
        originals=[]
        try:
            for source_key, group in h["fields"].items():
                attrs={}
                for name,value in group.attrs.items():
                    if isinstance(value,(str,bytes)):
                        value=_text(value)
                        if value[:1] in ("[","{"): value=json.loads(value)
                    elif isinstance(value,np.generic): value=value.item()
                    attrs[name]=value
                originals.append(dict(attrs,source_key=source_key))
                ds=group["values"]
                if ds.dtype.kind not in "fiu" or ds.ndim==0:
                    raise ValidationError("legacy fields must be numeric and frame-major")
                names=tuple(attrs.get("components",())) or ("scalar",)
                scalar_samples=attrs.get("provenance",{}).get("scalar_sample_axes")
                if ds.ndim==1:
                    declared_frames=tuple(attrs.get('frames',()))
                    if attrs.get('location')=='global' and declared_frames and len(declared_frames)==len(ds):
                        frame_count=len(ds);spatial_shape=();component_axis=False
                    else:
                        frame_count=1; spatial_shape=ds.shape; component_axis=False
                else:
                    frame_count=ds.shape[0]
                    if len(names)>1:
                        if ds.ndim<3 or ds.shape[-1]!=len(names):
                            raise ValidationError("ambiguous legacy component axis")
                        component_axis=True; spatial_shape=ds.shape[1:-1]
                    elif scalar_samples is not None:
                        if list(scalar_samples)!=list(range(2,ds.ndim)):
                            raise ValidationError("ambiguous legacy scalar sample axes")
                        component_axis=False; spatial_shape=ds.shape[1:]
                    elif ds.ndim>=3 and ds.shape[-1]==1:
                        component_axis=True; spatial_shape=ds.shape[1:-1]
                    else:
                        component_axis=False; spatial_shape=ds.shape[1:]
                field_frames=tuple(attrs.get('frames',()))
                if field_frames and len(field_frames)!=frame_count:raise ValidationError('legacy descriptor frame table disagrees with field rows')
                frame_map={i:i for i in range(frame_count)}
                frame_axis='root'
                scoped_ids=None
                if self._frames and field_frames:
                    if len(field_frames)==len(self._frames) and all(float(value)==root or (np.isnan(float(value)) and np.isnan(root)) for value,root in zip(field_frames,self._frames)):
                        frame_map={i:i for i in range(frame_count)}
                    else:
                        matches=[[i for i,root in enumerate(self._frames) if root==float(v)] for v in field_frames]
                        proven=all(len(indices)==1 for indices in matches) and len({indices[0] for indices in matches if indices})==frame_count
                        if proven:frame_map={indices[0]:local for local,indices in enumerate(matches)}
                        else:
                            # Independent quantity axes are not engineering
                            # root frames. Strong scoped identities name only
                            # their actual source rows, in original order.
                            frame_axis='quantity';frame_map={};scoped_ids=[]
                            for local,value in enumerate(field_frames):
                                cid=f'field:{source_key}:{local}'
                                token=-len(self._legacy_scoped_cases)-1
                                self._legacy_scoped_cases[cid]=(token,dict(case_id=cid,status='partial' if bool(h.attrs.get('partial',False)) else 'completed',certified=False,
                                    route='legacy_stored',quantity_id=source_key,frame_axis='quantity',local_frame_index=local,frame=float(value)))
                                scoped_ids.append(cid);frame_map[token]=local
                elif self._frames and frame_count!=len(self._frames):
                    if ds.ndim!=1:raise ValidationError('legacy partial field rows require explicit descriptor frames')
                    # An undeclared spatial vector has no proved root frame.
                    # Local stored row identity does not invent a time/load.
                    frame_axis='quantity';frame_map={};scoped_ids=[]
                    cid=f'field:{source_key}:0';token=-len(self._legacy_scoped_cases)-1
                    self._legacy_scoped_cases[cid]=(token,dict(case_id=cid,status='partial' if bool(h.attrs.get('partial',False)) else 'completed',certified=False,
                        route='legacy_stored',quantity_id=source_key,frame_axis='quantity',local_frame_index=0,frame=None,source_frame_declared=False))
                    scoped_ids.append(cid);frame_map[token]=0
                if not self._frames:maximum=max(maximum,frame_count)
                for ci, name in enumerate(names):
                    key=f"{source_key}::{name}"
                    d=dict(attrs,key=key,source_key=source_key,quantity_id=source_key,component=name,
                           sampling="legacy",availability="stored",component_shape=list(spatial_shape),stored_shape=list(ds.shape),
                           location=attrs.get("location","global"),unit=attrs.get("unit",""),basis=attrs.get("basis","global"))
                    d['frame_axis']=frame_axis;d['local_frame_indices']=list(range(frame_count))
                    d['frame_indices']=list(frame_map) if scoped_ids is None else None
                    d['case_ids']=[str(i) for i in frame_map] if scoped_ids is None else list(scoped_ids)
                    self._legacy_frame_maps[key]=frame_map
                    self._descriptors[key]=d; self._pairs[(source_key,name)]=key
                    self._counts[key]=int(np.prod(spatial_shape))
                    self._legacy_fields[key]=(source_key,ci,component_axis,spatial_shape,frame_count)
            partial=bool(h.attrs.get("partial",False))
            cases=[dict(case_id=str(i),status="partial" if partial else "completed",certified=False,
                        route="legacy_stored",frame_index=i,frame=self._frames[i] if self._frames else float(i)) for i in range(maximum)]
            provenance={}
            for name,ds in h.get("metadata",{}).items():
                provenance[name.removesuffix("_json")]=json.loads(_text(ds[()]))
            self._meta=dict(schema="anyfem.result",version=1,runtime=None,
                            model_identity=self.model_identity,mesh_identity=self.mesh_identity,
                            cases=cases,descriptors=list(self._descriptors.values()),patterns=[],
                            provenance=provenance,original_descriptors=originals,
                            root_attributes={k:_text(v) if isinstance(v,(str,bytes)) else v.item() if isinstance(v,np.generic) else v for k,v in h.attrs.items()},
                            frame_kind=_text(h.get("frames",h).attrs.get("kind","static")),
                            history_keys=list(h.get('histories',{}).keys()),
                            history_shapes={name:list(ds.shape) if hasattr(ds,'shape') else list(ds['x'].shape) for name,ds in h.get('histories',{}).items() if hasattr(ds,'shape') or ('x' in ds and 'y' in ds and hasattr(ds['x'],'shape') and hasattr(ds['y'],'shape') and ds['x'].shape==ds['y'].shape)},
                            table_shapes={name:list(ds.shape) for name,ds in h.get('tables',{}).items() if hasattr(ds,'shape')},
                            table_dtypes={name:str(ds.dtype) for name,ds in h.get('tables',{}).items() if hasattr(ds,'dtype')})
            self._case_index={c["case_id"]:i for i,c in enumerate(cases)}
        except (KeyError,ValueError,TypeError) as exc:
            if isinstance(exc,ValidationError): raise
            raise ValidationError(f"invalid ANYfem sidecar: {exc}") from exc
    def _case(self,case_id):
        self._check_cancel()
        scoped=self._legacy_scoped_cases.get(case_id) if isinstance(case_id,str) else None
        if scoped is None:return super()._case(case_id)
        index,case=scoped
        if case['status']!='completed':raise FailedCaseError(f'case {case_id!r}: {case["status"]}; see case_metadata() diagnostics')
        return index,case
    # Raw sidecar labels are source metadata, not native JSON. Keep copies
    # local to this compatibility reader; native archive cloning stays strict.
    def cases(self):
        with self._lock:
            self._ensure_open();return deepcopy(tuple(self._meta['cases']))
    def components(self,quantity_id=None):
        with self._lock:
            self._ensure_open();return deepcopy(tuple(d for d in self._meta['descriptors'] if quantity_id is None or d['quantity_id']==quantity_id))
    def metadata(self):
        with self._lock:
            self._ensure_open();return deepcopy(self._meta)
    def case_metadata(self,case_id):
        with self._lock:
            self._ensure_open()
            scoped=self._legacy_scoped_cases.get(case_id) if isinstance(case_id,str) else None
            if scoped is not None:return deepcopy(scoped[1])
            try:index=self._case_index[case_id]
            except (KeyError,TypeError):raise UnknownCaseError(f'unknown case {case_id!r}') from None
            return deepcopy(self._meta['cases'][index])
    def case_summaries(self):
        names=('case_id','status','certified','route','input_index','active_indices','frame_index','frame')
        with self._lock:
            self._ensure_open();return deepcopy(tuple({name:case[name] for name in names if name in case} for case in self._meta['cases']))
    def metadata_summary(self):
        with self._lock:
            self._ensure_open();result=deepcopy(self._meta);result['cases_complete']=True;return result
    def _admit(self, case_index, key, seen=None, *, runtime_required=True):
        self._check_cancel()
        if case_index not in self._legacy_frame_maps[key]:
            raise UnavailableFieldError("legacy field does not contain requested frame")
    def _cost(self,key):
        _,_,component_axis,shape,_=self._legacy_fields[key]
        return 64+8*(len(shape)+1)
    @_io_boundary
    def _block(self,case_index,key,indices,memo=None):
        self._check_cancel()
        source,component,component_axis,shape,frames=self._legacy_fields[key]
        if case_index not in self._legacy_frame_maps[key]:
            raise UnavailableFieldError("legacy field does not contain requested frame")
        case_index=self._legacy_frame_maps[key][case_index]
        ds=self._handle[f"fields/{source}/values"]
        result=np.empty(len(indices),dtype=np.float64)
        # HDF5 point reads preserve arbitrary spatial sample order without
        # materializing all components or a complete frame.
        if ds.ndim==1:
            for j,index in enumerate(indices):
                self._check_cancel(); result[j]=ds[case_index] if shape==() else ds[int(index)]
        elif len(shape)==1 and isinstance(indices,range) and indices.step==1:
            sl=slice(indices.start,indices.stop)
            result[:]=ds[case_index,sl,component] if component_axis else ds[case_index,sl]
        else:
            for j,index in enumerate(indices):
                self._check_cancel()
                coords=tuple(int(v) for v in np.unravel_index(int(index),shape))
                address=(case_index,)+coords+((component,) if component_axis else ())
                result[j]=ds[address]
        self._reads+=1
        # NaN is ANYfem's documented missing-entity marker; retain it verbatim.
        return result
    @_io_boundary
    @_cancellable
    def associations(self,quantity_id,component,selection=None, *, cancel_event=None):
        with self._lock:
            self._ensure_open(); key=self._key(quantity_id,component)
            source,_,_,shape,_=self._legacy_fields[key]
            location=self._descriptors[key]["location"]
            table_name=f"{source}_{'node_ids' if location=='node' else 'element_ids'}"
            if table_name not in self._handle.get("tables",{}):
                raise UnavailableFieldError("legacy field has no explicit entity association")
            table=self._handle["tables"][table_name]
            if table.dtype.kind not in "iu" or table.ndim not in (1,2) or len(table)!=shape[0]:
                raise ValidationError("legacy entity association disagrees with field")
            indices,_=self._selection(self._counts[key],selection)
            samples=int(np.prod(shape[1:])) if len(shape)>1 else 1
            width=(1 if table.ndim==1 else table.shape[1])+(len(shape)-1)
            self._reserve(len(indices)*(width*16+64))
            result=np.empty((len(indices),width),dtype=np.int64)
            for j,index in enumerate(indices):
                self._check_cancel()
                entity=int(index)//samples
                value=np.asarray(table[entity]).reshape(-1)
                result[j,:len(value)]=value
                if len(shape)>1: result[j,len(value):]=np.unravel_index(int(index)%samples,shape[1:])
            self._reads+=1
            return _freeze(result[:,0] if width==1 else result)
    @_cancellable
    def geometry(self,selection=None, *, cancel_event=None):
        # ANYfem result sidecars reference a separately owned mesh sidecar.
        with self._lock: self._ensure_open(); return {}
    @_io_boundary
    def table_keys(self):
        with self._lock: self._ensure_open(); return tuple(self._handle.get("tables",{}).keys())
    @_io_boundary
    def table(self,key,rows=None):
        with self._lock:
            self._ensure_open(); ds=self._handle["tables"][key]
            if ds.ndim==0:
                raw=ds[()]; self._reads+=1
                return json.loads(_text(raw)) if isinstance(raw,(str,bytes)) else raw
            indices,_=self._selection(len(ds),rows)
            self._reserve(len(indices)*int(np.prod(ds.shape[1:] or (1,)))*ds.dtype.itemsize*3+len(indices)*32)
            return _freeze(self._spatial_read(ds,indices))
    def frames(self):
        with self._lock: self._ensure_open(); return _freeze(np.asarray(self._frames,dtype=np.float64))

    @_io_boundary
    @_cancellable
    def read_history(self,key,selection=None,*,rows=None,cancel_event=None):
        """Select the frame axis of an existing numeric ANYfem history."""
        with self._lock:
            self._ensure_open()
            if rows is not None:
                if selection is not None:raise TypeError('supply selection or rows, not both')
                selection=rows
            if not isinstance(key,str) or not key or '/' in key:raise ValidationError('history key must be one dataset name')
            histories=self._handle.get('histories')
            if histories is None or key not in histories:raise UnavailableFieldError('legacy history is unavailable: '+key)
            entry=histories[key]
            if hasattr(entry,'dtype'):datasets=(entry,)
            else:
                if 'x' not in entry or 'y' not in entry:raise ValidationError('legacy history group requires x/y datasets')
                datasets=(entry['x'],entry['y'])
            if any(not hasattr(ds,'dtype') or ds.ndim<1 or ds.dtype.kind not in 'fiu' for ds in datasets):raise ValidationError('legacy histories require numeric frame-major arrays')
            if len(datasets)==2 and datasets[0].shape!=datasets[1].shape:raise ValidationError('legacy history x/y shapes disagree')
            indices,_identity=self._selection(len(datasets[0]),selection)
            index_bytes=len(indices)*16 if isinstance(indices,np.ndarray) else 0
            row_bytes=sum(int(np.prod(ds.shape[1:] or (1,)))*ds.dtype.itemsize for ds in datasets)
            self._reserve(len(indices)*row_bytes*3+index_bytes+len(indices)*32)
            results=[]
            for dataset in datasets:
                value=np.empty((len(indices),)+dataset.shape[1:],dtype=dataset.dtype)
                for position,index in enumerate(indices):
                    self._check_cancel();value[position]=dataset[int(index)]
                self._reads+=len(indices)
                self._check_cancel()
                results.append(np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape))
            return results[0] if len(results)==1 else tuple(results)

    @_io_boundary
    @_cancellable
    def read_stored_field(self,case_id,quantity_id,component,selection=None,*,cancel_event=None):
        """Explicit stored-only access; incomplete analysis status is preserved."""
        with self._lock:
            self._ensure_open()
            scoped=self._legacy_scoped_cases.get(case_id) if isinstance(case_id,str) else None
            if scoped is not None:index=scoped[0]
            else:
                try:index=self._case_index[case_id]
                except (KeyError,TypeError):raise UnknownCaseError(case_id) from None
            key=self._key(quantity_id,component);self._admit(index,key)
            indices,_=self._selection(self._counts[key],selection)
            self._reserve(len(indices)*(self._cost(key)+40))
            value=self._block(index,key,indices)
            return np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)
