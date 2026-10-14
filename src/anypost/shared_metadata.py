"""Admitted native-v2 fixed case metadata; exact JSON values, no numeric coercion."""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import re
import sys
import numpy as np
from .errors import ValidationError, WorkingSetError
from .validation import json_text

CODEC = "reaction_node_ids/1"
FIELD = "reaction_node_ids"
GROUP = "shared_case_metadata"
MIN_IDS = 256
BLOCK_BYTES = 256 * 1024


def clone(value):
    return json.loads(json_text(value))


def summary(record):
    # Explicit bounded listing: authoritative case identity/status/route and small
    # control fields only. Detailed diagnostic/solver payloads remain queryable.
    names = ("case_id", "status", "certified", "route", "input_index", "active_indices", "frame_index", "frame")
    return {name: clone(record[name]) for name in names if name in record}


def _detail_size(value, budget):
    used=sys.getsizeof(value)
    if used>budget:return used
    if isinstance(value,dict):
        for key,item in value.items():
            used+=_detail_size(key,budget-used)+_detail_size(item,budget-used)
            if used>budget:return used
    elif isinstance(value,(list,tuple)):
        for item in value:
            used+=_detail_size(item,budget-used)
            if used>budget:return used
    return used


def _small_details(value, budget):
    # Large structures are refused by shallow size before visiting elements.
    return _detail_size(value,budget)<=budget


def load_native_metadata(text,version):
    if version==1:return json.loads(text)
    def object_pairs(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValidationError("duplicate native-v2 metadata JSON key")
            value[key]=item
        return value
    return json.loads(text,object_pairs_hook=object_pairs)


class CaseMetadataWriter:
    def __init__(self, handle, model_identity, mesh_identity, compression):
        self.handle = handle
        self.model_identity = model_identity
        self.mesh_identity = mesh_identity
        self.compression = compression
        self.references = {}
        self.tables = {}
        self._digests = {}
        self.peak_input_bytes = 0

    def add_case(self, record, index):
        if not isinstance(record, Mapping):
            raise ValidationError("case metadata must be a mapping")
        ids = record.get(FIELD)
        if not isinstance(ids, list) or len(ids) < MIN_IDS or any(type(value) is not int for value in ids):
            return clone(record)
        payload = json_text(ids).encode("utf-8")
        self.peak_input_bytes = max(self.peak_input_bytes, len(payload))
        digest = hashlib.sha256(payload).hexdigest()
        table = self._digests.get(digest)
        if table is None:
            table = str(len(self.tables))
            group = self.handle.require_group(GROUP)
            ds = group.create_dataset(table,shape=(len(payload),),dtype="uint8",chunks=(min(BLOCK_BYTES,len(payload)),),compression=self.compression,shuffle=True,fletcher32=True)
            for start in range(0,len(payload),BLOCK_BYTES):
                ds[start:start+BLOCK_BYTES] = np.frombuffer(payload,dtype=np.uint8,count=min(BLOCK_BYTES,len(payload)-start),offset=start)
            self.tables[table] = {"sha256":digest,"bytes":len(payload),"count":len(ids)}
            self._digests[digest] = table
        else:
            ds = self.handle[GROUP][table]
            if len(payload) != len(ds):raise ValidationError("shared metadata digest collision")
            for start in range(0,len(payload),BLOCK_BYTES):
                if ds[start:start+BLOCK_BYTES].tobytes() != payload[start:start+BLOCK_BYTES]:
                    raise ValidationError("shared metadata digest collision")
        compact = clone({key:value for key,value in record.items() if key != FIELD})
        compact[FIELD] = None
        self.references[str(index)] = table
        return compact

    def finish(self, metadata):
        if not self.references:return 1
        metadata[GROUP] = {"codec":CODEC,"path":[FIELD],"model_identity":self.model_identity,"mesh_identity":self.mesh_identity,"references":dict(self.references),"tables":dict(self.tables)}
        return 2


class CaseMetadataReader:
    def __init__(self, handle, metadata, version, working_bytes):
        self.handle = handle
        self.metadata = metadata
        self.working_bytes = working_bytes
        self.references = {}
        self.tables = {}
        self.verified_table_bytes = 0
        self.small_details_complete = version==1 and _small_details(metadata["cases"],min(64*1024,working_bytes//8))
        if version == 1:
            if GROUP in metadata or (handle is not None and GROUP in handle):
                raise ValidationError("shared case metadata requires native version 2")
            return
        container = metadata.get(GROUP)
        if not isinstance(container,dict) or set(container)!={"codec","path","model_identity","mesh_identity","references","tables"}:
            raise ValidationError("native version 2 requires admitted shared case metadata")
        if container["codec"] != CODEC or container["path"] != [FIELD]:
            raise ValidationError("unsupported shared case metadata codec/path")
        if any(container[name]!=metadata[name] for name in ("model_identity","mesh_identity")):
            raise ValidationError("shared case metadata source identity mismatch")
        refs,tables=container["references"],container["tables"]
        if not isinstance(refs,dict) or not refs or not isinstance(tables,dict) or not tables or GROUP not in handle:
            raise ValidationError("missing shared case metadata references/tables")
        group=handle[GROUP]
        if set(group)!=set(tables):raise ValidationError("shared case table inventory mismatch")
        cases=metadata["cases"]
        for index,table in refs.items():
            if not isinstance(index,str) or not index.isdigit() or str(int(index))!=index or int(index)>=len(cases) or not isinstance(table,str) or table not in tables:
                raise ValidationError("dangling shared case metadata reference")
            case=cases[int(index)]
            if FIELD not in case or case[FIELD] is not None:
                raise ValidationError("shared case metadata placeholder/path mismatch")
        if set(refs.values())!=set(tables):raise ValidationError("unreferenced shared case metadata table")
        self.references=refs;self.tables=tables
        # Admit each table once without retaining decoded ID lists. Summaries and
        # engineering replay keep compact case records; explicit detail reads clone.
        for table in tables:self._verify_table(table)

    def _dataset(self,table):
        spec=self.tables[table]
        if not isinstance(spec,dict) or set(spec)!={"sha256","bytes","count"} or any(type(spec[name]) is not int or spec[name]<=0 for name in ("bytes","count")) or not isinstance(spec["sha256"],str) or re.fullmatch("[0-9a-f]{64}",spec["sha256"]) is None:
            raise ValidationError("invalid shared case metadata table descriptor")
        if not isinstance(table,str) or not table.isdigit() or str(int(table))!=table:
            raise ValidationError("invalid shared case metadata table name")
        ds=self.handle[GROUP][table]
        if not hasattr(ds,"dtype") or ds.ndim!=1 or ds.dtype!=np.dtype("uint8") or ds.shape!=(spec["bytes"],):
            raise ValidationError("invalid shared case metadata table dataset")
        return ds,spec

    def _verify_table(self,table):
        ds,spec=self._dataset(table)
        digest=hashlib.sha256();count=0;token=bytearray();position=0;finished=False
        step=min(BLOCK_BYTES,max(1,self.working_bytes//4))
        for start in range(0,len(ds),step):
            block=ds[start:start+step].tobytes();digest.update(block)
            for char in block:
                if position==0:
                    if char!=91:raise ValidationError("shared reaction IDs require a canonical integer JSON list")
                elif finished:
                    raise ValidationError("trailing shared case metadata JSON")
                elif char in (44,93):
                    if not token or re.fullmatch(rb"-?(0|[1-9][0-9]*)",token) is None or token==b"-0":
                        raise ValidationError("shared reaction IDs require canonical plain integers")
                    count+=1;token.clear();finished=char==93
                else:
                    token.append(char)
                    if len(token)>step:raise WorkingSetError("shared metadata integer token exceeds working budget")
                position+=1
        if not finished or token or count!=spec["count"] or digest.hexdigest()!=spec["sha256"]:
            raise ValidationError("shared case metadata digest/count/JSON mismatch")
        self.verified_table_bytes+=len(ds)

    def detail_work(self,index):
        base=_detail_size(self.metadata["cases"][index],self.working_bytes//8)*8+512
        table=self.references.get(str(index))
        if table is not None:
            spec=self.tables[table]
            base+=spec["bytes"]*8+spec["count"]*48+BLOCK_BYTES
        return base

    def require_detail_work(self,indices):
        if not self.references:return
        if sum(self.detail_work(index) for index in indices)>self.working_bytes:
            raise WorkingSetError("explicit case detail expansion exceeds working budget; use case_summaries or one case_metadata query")

    def _read_table(self, table):
        ds,spec=self._dataset(table)
        payload=bytearray(spec["bytes"]);digest=hashlib.sha256()
        for start in range(0,len(payload),BLOCK_BYTES):
            block=ds[start:start+BLOCK_BYTES].tobytes();digest.update(block);payload[start:start+len(block)]=block
        if digest.hexdigest()!=spec["sha256"]:raise ValidationError("shared case metadata digest mismatch")
        try:value=json.loads(payload)
        except (ValueError,UnicodeError) as exc:raise ValidationError("invalid shared case metadata JSON") from exc
        if not isinstance(value,list) or len(value)!=spec["count"] or any(type(item) is not int for item in value):
            raise ValidationError("shared reaction IDs must preserve a plain integer list")
        return value

    def detail(self,index):
        self.require_detail_work((index,))
        value=clone(self.metadata["cases"][index])
        table=self.references.get(str(index))
        if table is not None:value[FIELD]=self._read_table(table)
        return value

    def summaries(self):
        return tuple(summary(record) for record in self.metadata["cases"])

    def public_metadata(self,*,details):
        if details:self.require_detail_work(range(len(self.metadata["cases"])))
        result=clone({key:value for key,value in self.metadata.items() if key not in {"cases",GROUP}})
        include_details=details or self.small_details_complete
        result["cases"]=[self.detail(index) for index in range(len(self.metadata["cases"]))] if include_details else list(self.summaries())
        if not details:result["cases_complete"]=self.small_details_complete
        return result
