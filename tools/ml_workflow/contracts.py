"""Validated experiment and chronological case contracts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DAY = 86400
JAN20 = 1737331200
V6_SHA = "c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c"
MEMBERS = {"dim8_seed42":(8,42,260), "dim8_seed43":(8,43,260), "dim8_seed44":(8,44,260),
           "dim16_seed42":(16,42,196), "dim16_seed43":(16,43,196)}
STAGES = ("develop", "screen", "stress", "finalize")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def identifier(value):
    require(isinstance(value,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",value), "Unsafe identifier")
    return value


def inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe relative path")
    path = root/relative
    current = root
    for part in relative.parts:
        current = current/part
        require(not current.is_symlink(), "Symlink in protected path")
    require(path.resolve().is_relative_to(root), "Path escapes root")
    return path


@dataclass(frozen=True)
class CaseSpec:
    name: str
    meta_cutoff: int
    target_cutoff: int
    target_end: int
    baseline: str

    @classmethod
    def parse(cls, value):
        result = cls(**value)
        identifier(result.name); identifier(result.baseline)
        require(all(type(v) is int for v in (result.meta_cutoff,result.target_cutoff,result.target_end)), "Cutoffs must be integer UTC seconds")
        require(result.meta_cutoff+DAY <= result.target_cutoff < result.target_end, "Case must transfer between chronological neural prefixes")
        return result


@dataclass(frozen=True)
class ExperimentSpec:
    value: dict

    @classmethod
    def parse(cls, value):
        require(value.get("schema_version") in (1,2,3), "Unsupported specification version")
        post=value["schema_version"] in (2,3)
        if post:require(value.get("mode")=="ensemble_postprocessor", "Unsupported schema-v2 mode")
        require(value.get("purpose") in {"synthetic_validation","research"}, "Unknown experiment purpose")
        require(isinstance(value.get("hypothesis"),str) and value["hypothesis"].strip(), "Declare one hypothesis")
        require(value.get("comparator")=="V6", "Comparator must be V6")
        adapter = Path(value["adapter"])
        require(not adapter.is_absolute() and ".." not in adapter.parts and adapter.parts[0]=="experiments" and adapter.suffix==".py", "Adapter must be repository experiment source")
        candidates = value["candidates"]
        require(bool(candidates) and isinstance(candidates,dict), "Declare candidates before development")
        for name, params in candidates.items():
            identifier(name)
            if post:
                require(params.get("kind")=="ensemble_postprocessor", "Invalid postprocessor candidate")
            else:
                changed=params["members"]
                require(bool(changed) and len(set(changed))==len(changed) and set(changed)<=set(MEMBERS), "Unknown or repeated changed members")
        order=value["candidate_order"]
        require(isinstance(order,list) and len(order)==len(candidates) and set(order)==set(candidates),"Declare an explicit candidate tie order")
        require(type(value.get("seed")) is int, "Declare deterministic seed")
        require(value.get("bootstrap")=={"resamples":1000,"seed":2026,"minimum_valid":990}, "Promotion bootstrap contract changed")
        require(value.get("selection")==("maximin_nonnegative_origins_and_late" if value["schema_version"]==3 else "maximin_nonnegative_origins"), "Unsupported selection rule")
        if value["schema_version"]==3:
            require(value.get("postprocessing_contract")=={"event_mass":"learned","within_event_order":"preserve","state_schema_version":2}, "Invalid schema-v3 postprocessing contract")
        for name, asset in value["inputs"].items():
            identifier(name)
            require(isinstance(asset,dict) and re.fullmatch(r"[a-f0-9]{64}",asset["sha256"]), "Invalid input fingerprint")
            require(not Path(asset["path"]).is_absolute() and ".." not in Path(asset["path"]).parts, "Unsafe input path")
        require(({"train","evaluation","inventory"} if post else {"train","evaluation","baseline_evaluation"}) <= set(value["inputs"]), "Missing event/baseline inputs")
        cases = [CaseSpec.parse(c) for c in value["development"]]
        require(cases and len({c.name for c in cases})==len(cases), "Development cases must be unique and nonempty")
        require(all(c.target_end<=JAN20+5*DAY for c in cases), "Development must end before Jan25")
        if value["schema_version"]==3:
            require(any(c.target_end>c.target_cutoff+2*DAY for c in cases), "Missing aged development case")
        expected={"primary":(JAN20+4*DAY,JAN20+5*DAY,JAN20+7*DAY),
                  "secondary":(JAN20+5*DAY,JAN20+6*DAY,JAN20+7*DAY),
                  "five_day":(JAN20+DAY,JAN20+2*DAY,JAN20+7*DAY)}
        require(set(value["promotion"])==set(expected), "Incomplete promotion cases")
        for name, dates in expected.items():
            case=CaseSpec.parse(value["promotion"][name])
            require(case.name==name and (case.meta_cutoff,case.target_cutoff,case.target_end)==dates, "Historical promotion chronology changed")
        for case in cases+[CaseSpec.parse(c) for c in value["promotion"].values()]:
            require(post or case.baseline in value["inputs"], "Missing explicit baseline input")
        budget=value["budget"]
        require(type(budget["phase_seconds"]) is int and 0<budget["phase_seconds"]<=14400, "Invalid phase cap")
        require(type(budget["total_seconds"]) is int and 0<budget["total_seconds"]<=14400, "Invalid global cap")
        if value["purpose"]=="research":
            require(budget["total_seconds"]==14400 and "carry" in value["inputs"], "Research requires original budget and a verified carry decision")
            require(identifier(value["release_name"]) not in {"v6","v7"}, "Protected release name")
            require(value.get("device") in {"cpu","cuda"},"Declare research device")
        require(value.get("baseline_sha256")==V6_SHA, "V6 baseline identity changed")
        return cls(json.loads(json.dumps(value,allow_nan=False)))


def first_indices(users,times,begin,end):
    eligible=np.flatnonzero((times>=begin)&(times<end))
    ordered=eligible[np.argsort(times[eligible],kind="stable")]
    _, positions=np.unique(users[ordered],return_index=True)
    return np.sort(ordered[positions])


def validate_events(data, labeled):
    required={"u","i","t","tb","dt","h","ht","rowids"} | ({"y"} if labeled else set())
    require(required<=set(data), "Incomplete event schema")
    if not labeled:
        require(not ({"y","clicked","label","labels","target"}&set(data)), "Evaluation labels forbidden")
    require(set(data)==required,"Unexpected event fields")
    n=len(data["u"])
    require(n and all(data[k].shape==(n,) for k in ("u","t","tb","dt")), "Invalid event vectors")
    require(data["u"].dtype.kind in "iu" and (data["u"]>=0).all(), "Invalid user IDs")
    require(data["t"].dtype.kind in "iu" and data["ht"].dtype.kind in "iu", "Timestamps must be integer UTC seconds")
    require(data["i"].shape==(n,10) and data["i"].dtype.kind in "iu" and ((data["i"]>=0)&(data["i"]<50000)).all(), "Invalid candidates")
    require(data["h"].shape==data["ht"].shape==(n,20), "Invalid history shape")
    require(np.isin(data["tb"],[0,1,2,3]).all() and np.isin(data["dt"],[0,1]).all(), "Invalid event context")
    require(data["rowids"].shape==(n,10) and data["rowids"].dtype.kind=="U"
        and len(np.unique(data["rowids"]))==n*10, "Invalid or repeated row IDs")
    present=data["h"]>=0
    require(data["h"].dtype.kind in "iu" and ((data["h"]>=-1)&(data["h"]<50000)).all()
        and (data["ht"][present]<np.broadcast_to(data["t"][:,None],present.shape)[present]).all(), "History crosses event timestamp")
    if labeled:
        require(data["y"].shape==(n,10) and np.isin(data["y"],[0,1]).all() and (data["y"].sum(1)<=1).all(), "Invalid click labels")


def context(data,indices,labeled):
    return {k:v[indices].copy() for k,v in data.items() if labeled or k not in {"y","clicked","label","labels","target"}}


def compose(members):
    require(set(members)==set(MEMBERS), "Incomplete ensemble")
    a,b,c,d,e=[np.asarray(members[m],dtype=np.float64) for m in MEMBERS]
    require(all(v.shape==a.shape and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in (a,b,c,d,e)), "Invalid member probabilities")
    return (2/3)*((a+b+c)/3)+(1/3)*(.5*d+.5*e)
