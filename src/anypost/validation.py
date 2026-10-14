"""Capsule validation independent of any solver installation."""
import json
from collections.abc import Mapping
import numpy as np
from .errors import ValidationError

SUCCESS = frozenset(("completed", "complete", "success", "succeeded", "ok", "converged"))
DESCRIPTOR_FIELDS = ("key", "quantity_id", "component", "location", "unit", "basis", "sampling", "availability")

def json_text(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"metadata must contain only finite JSON data: {exc}") from exc

def array(value, name, *, integer=False, boolean=False, ndim=None):
    a = np.asarray(value)
    allowed = "iu" if integer else "fiub" if boolean else "fiu"
    if a.dtype.kind not in allowed or (ndim is not None and a.ndim != ndim):
        raise ValidationError(f"{name}: invalid numeric dtype or dimension")
    # Finite checks are bounded, including non-contiguous caller views.
    it = np.nditer(a, flags=["external_loop", "buffered", "zerosize_ok"], op_flags=["readonly"], buffersize=32768)
    for block in it:
        if not np.isfinite(block).all():
            raise ValidationError(f"{name}: nonfinite engineering data")
    if integer and a.dtype.kind == "u" and a.size and np.max(a) > np.iinfo(np.int64).max:
        raise ValidationError(f"{name}: integer exceeds portable int64 range")
    return a

def replay_contract(meta, *, capsule=False):
    version = meta.get("version") if capsule else meta.get("capsule_version")
    runtime = meta.get("runtime")
    if version is None and not capsule and runtime == "anysolver-results/1":
        version = 1
    if type(version) is not int or (version, runtime) not in ((1,"anysolver-results/1"),(2,"anysolver-results/2")):
        raise ValidationError("unsupported paired capsule version/runtime identity")
    combination = meta.get("combination", "legacy_blas/1" if version == 1 else None)
    if combination != ("legacy_blas/1" if version == 1 else "ordered_float64/1"):
        raise ValidationError("unsupported paired recovery combination")
    return version, combination


def validate_metadata(meta, *, allow_unknown_case_components=False):
    for name in ("model_identity", "mesh_identity"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValidationError(f"{name} must be a nonempty string")
    capsule_version, _ = replay_contract(meta)
    patterns = meta.get("patterns", ())
    if any(not isinstance(p, str) or not p for p in patterns) or len(set(patterns)) != len(patterns):
        raise ValidationError("pattern names must be unique nonempty strings")
    cases, descriptors = meta.get("cases", ()), meta.get("descriptors", ())
    ids = set()
    for case in cases:
        if not isinstance(case, Mapping) or any(k not in case for k in ("case_id", "status", "certified", "route")):
            raise ValidationError("case is missing identity/status/certification/route")
        cid = case["case_id"]
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValidationError("case IDs must be unique nonempty strings")
        ids.add(cid)
        if type(case["certified"]) is not bool or not isinstance(case["status"], str) or not isinstance(case["route"], str):
            raise ValidationError("case status, certified and route have invalid types")
        if case["certified"] and case["status"] not in SUCCESS:
            raise ValidationError("failed or incomplete case cannot be certified")
        if capsule_version == 2 and any(name not in case for name in ("portable","displacement_portable")):
            raise ValidationError("capsule2 case requires portable and displacement_portable flags")
        if "portable" in case and type(case["portable"]) is not bool:
            raise ValidationError("portable must be a boolean")
        if "displacement_portable" in case and case["displacement_portable"] is not None and type(case["displacement_portable"]) is not bool:
            raise ValidationError("displacement_portable must be a boolean or null")
        missing = case.get("unavailable_components", [])
        if not isinstance(missing, (list,tuple)) or any(not isinstance(k,str) or not k for k in missing) or len(set(missing)) != len(missing):
            raise ValidationError("unavailable_components must contain unique component keys")
        active = case.get("active_indices")
        if active is not None and (not isinstance(active, (list, tuple)) or any(type(i) is not int or i < 1 or i > len(patterns) for i in active) or len(set(active)) != len(active)):
            raise ValidationError("invalid original active pattern indices")
    keys, pairs = set(), set()
    for d in descriptors:
        if not isinstance(d, Mapping) or any(k not in d for k in DESCRIPTOR_FIELDS):
            raise ValidationError("descriptor is missing required metadata")
        if any(not isinstance(d[k], str) for k in DESCRIPTOR_FIELDS):
            raise ValidationError("descriptor metadata must be strings")
        pair = (d["quantity_id"], d["component"])
        if not d["key"] or d["key"] in keys or pair in pairs:
            raise ValidationError("descriptor keys and quantity/component pairs must be unique")
        keys.add(d["key"]); pairs.add(pair)
        if d["availability"] not in ("stored", "reconstructable", "unavailable"):
            raise ValidationError("unsupported descriptor availability")
        if capsule_version==2 and d["availability"]=="reconstructable":
            # Source2 wire classes are closed independently of a caller label.
            # Stored custom fields and legacy1 keep their existing meaning.
            nodal={"displacement":("UX","UY","UZ","RX","RY","RZ"),"reaction":("FX","FY","FZ","MX","MY","MZ")}
            reserved=d["quantity_id"] in nodal or d["key"].startswith(("displacement:","reaction:")) or any(d["component"] in components for components in nodal.values())
            if d["location"] not in ("node","element"):
                raise ValidationError("capsule2 replay requires canonical node/element location")
            if d["location"]=="node" or reserved:
                if d["location"]!="node" or d["quantity_id"] not in nodal or d["component"] not in nodal[d["quantity_id"]] or d["key"]!=f"{d['quantity_id']}:{d['component']}":
                    raise ValidationError("capsule2 nodal replay requires canonical source key, quantity, component and location")
            elif d["quantity_id"]!="stress" and not d["quantity_id"]==d["component"]==d["key"]:
                raise ValidationError("capsule2 element replay requires a canonical source wire class")
    if not allow_unknown_case_components and any(set(case.get("unavailable_components", ())) - keys for case in cases):
        raise ValidationError("unknown unavailable case component")
    policies = [d.get("tie_policy", "first") for d in descriptors]
    if any(not isinstance(policy, str) or policy not in {"first", "last", "exact_earliest_input_index"} for policy in policies):
        raise ValidationError("unsupported authoritative envelope tie policy")
    if "exact_earliest_input_index" in policies:
        indices = [case.get("input_index") for case in cases]
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValidationError("authoritative envelope policy requires unique nonnegative original input_index")
    derived = meta.get("derived", {})
    if not isinstance(derived, Mapping) or set(derived) - keys:
        raise ValidationError("unknown derived descriptor")
    for key, spec in derived.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("operation"), str) or not isinstance(spec.get("inputs"), Mapping):
            raise ValidationError("derived specification needs operation and inputs")
        if not spec["inputs"] or any(not isinstance(k, str) or v not in keys for k, v in spec["inputs"].items()):
            raise ValidationError("derived specification has unknown input")
    visiting, done = set(), set()
    def visit(key):
        if key in visiting:
            raise ValidationError("derived dependency cycle")
        if key in done:
            return
        visiting.add(key)
        for dep in derived.get(key, {}).get("inputs", {}).values():
            visit(dep)
        visiting.remove(key); done.add(key)
    for key in keys:
        visit(key)
    json_text(meta)


def validate_capsule(capsule):
    if not isinstance(capsule, Mapping) or capsule.get("schema") != "anysolver.recovery_capsule":
        raise ValidationError("expected recovery capsule")
    capsule_version, combination = replay_contract(capsule, capsule=True)
    required = ("solver_identity", "geometry", "patterns", "coefficients", "cases", "descriptors", "associations", "responses", "stored", "derived")
    if any(k not in capsule for k in required):
        raise ValidationError("incomplete recovery capsule")
    for name in ("solver_identity", "geometry", "associations", "responses", "stored", "derived"):
        if not isinstance(capsule[name], Mapping):
            raise ValidationError(f"{name} must be a mapping")
    meta = {k: capsule[k] for k in ("runtime", "model_identity", "mesh_identity", "solver_identity", "patterns", "cases", "descriptors", "derived")}
    meta.update(capsule_version=capsule_version, combination=combination)
    validate_metadata(meta)
    json_text({k: v for k, v in capsule.items() if k not in {"geometry", "coefficients", "associations", "responses", "stored"}})
    c = array(capsule["coefficients"], "coefficients", ndim=2)
    if c.shape != (len(meta["cases"]), len(meta["patterns"])):
        raise ValidationError("coefficient table shape mismatch")
    descriptors = {d["key"]: d for d in meta["descriptors"]}
    for name in ("associations", "responses", "stored"):
        if set(capsule[name]) - descriptors.keys():
            raise ValidationError(f"unknown {name} key")
    counts = {}
    for key, d in descriptors.items():
        if d["availability"] == "unavailable" and (key in capsule["responses"] or key in capsule["stored"] or key in capsule["derived"]):
            raise ValidationError("unavailable field contains success data")
        if key not in capsule["associations"]:
            if d["availability"] != "unavailable":
                raise ValidationError(f"{key}: readable field needs associations")
            counts[key] = 0
            continue
        a = array(capsule["associations"][key], f"associations/{key}", integer=True)
        if a.ndim not in (1, 2):
            raise ValidationError("association table must have one or two axes")
        counts[key] = len(a)
        if d["availability"] == "unavailable" and (key in capsule["responses"] or key in capsule["stored"] or key in capsule["derived"]):
            raise ValidationError("unavailable field contains success data")
    for key, value in capsule["responses"].items():
        a = array(value, f"responses/{key}", ndim=2)
        if a.shape != (counts[key], len(meta["patterns"]) + 1):
            raise ValidationError("response matrix shape mismatch")
        if descriptors[key]["availability"] != "reconstructable":
            raise ValidationError("response requires reconstructable descriptor")
    for key, entry in capsule["stored"].items():
        if not isinstance(entry, Mapping) or not {"case_indices", "values"} <= entry.keys():
            raise ValidationError("stored entry needs rows and values")
        ids = array(entry["case_indices"], "stored case indices", integer=True, ndim=1)
        values = array(entry["values"], "stored values", ndim=2)
        if values.shape != (len(ids), counts[key]) or np.any(ids < 0) or np.any(ids >= len(meta["cases"])) or len(np.unique(ids)) != len(ids):
            raise ValidationError("invalid stored row indices or shape")
    for key, d in descriptors.items():
        if d["availability"] == "reconstructable" and key not in capsule["responses"] and key not in capsule["derived"]:
            raise ValidationError("reconstructable field has no response or derived path")
        if d["availability"] == "stored" and key not in capsule["stored"]:
            raise ValidationError("stored descriptor has no stored rows")
        for dep in capsule["derived"].get(key, {}).get("inputs", {}).values():
            if counts[dep] != counts[key] or not np.array_equal(capsule["associations"][dep], capsule["associations"][key]):
                raise ValidationError("derived dependency associations differ")
    for key in capsule["derived"]:
        if descriptors[key]["availability"] != "reconstructable":
            raise ValidationError("derived recovery requires reconstructable descriptor")
    for name, value in capsule["geometry"].items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValidationError("invalid geometry array name")
        a = array(value, f"geometry/{name}", boolean=True,
                  integer=name in ("triangles", "lines", "point_indices", "triangle_to_element", "node_ids", "element_ids"))
        if name in ("positions", "normals") and (a.ndim != 2 or a.shape[1] != 3):
            raise ValidationError(f"geometry/{name}: expected (count, 3)")
        if name in ("triangles", "lines") and (a.ndim != 2 or a.shape[1] != (3 if name == "triangles" else 2)):
            raise ValidationError(f"geometry/{name}: invalid topology shape")
        if name in ("node_ids", "element_ids", "triangle_to_element", "point_indices") and a.ndim != 1:
            raise ValidationError(f"geometry/{name}: expected a one-dimensional ID/index table")
    geometry = capsule["geometry"]
    if "positions" in geometry:
        count = len(geometry["positions"])
        if "node_ids" in geometry and len(geometry["node_ids"]) != count:
            raise ValidationError("geometry node ID count differs from positions")
        for name in ("triangles", "lines", "point_indices"):
            if name in geometry:
                indices = np.asarray(geometry[name])
                if np.any(indices < 0) or np.any(indices >= count):
                    raise ValidationError(f"geometry/{name}: point index outside positions")
    if "triangle_to_element" in geometry and ("triangles" not in geometry or len(geometry["triangle_to_element"]) != len(geometry["triangles"])):
        raise ValidationError("geometry triangle association count mismatch")
    if capsule_version == 2:
        if "node_ids" in geometry and "positions" not in geometry:
            raise ValidationError("geometry node IDs require positions")
        for name, topology in (("line_to_element","lines"),("triangle_to_element","triangles")):
            if name not in geometry: continue
            ids = geometry.get("element_ids")
            mapping = array(geometry[name], "geometry/"+name, integer=True, ndim=1)
            if ids is None or topology not in geometry or len(mapping) != len(geometry[topology]) or np.any(mapping < 0) or np.any(mapping >= len(ids)):
                raise ValidationError("geometry dense element mapping requires matching topology and element IDs")
        if any(name in geometry for name in ("triangles","lines","point_indices")) and "positions" not in geometry:
            raise ValidationError("geometry topology requires positions")
    return meta
