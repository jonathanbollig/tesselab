"""
The disk cache of nodes marked cache = "disk": their outputs in <project>/tesselab/cache/<fingerprint>/, one .npz
file of arrays per output plus meta.json (the datasets' structure and attributes, each output's content hash,
diagnostics, time). It holds derived data only: deleting it never loses work, the outputs are computed again. A
fingerprint changes whenever anything the result depends on changes (engine.py), so an entry is never outdated, only
unused.

Why .npz and JSON rather than netCDF or pickle: the arrays come back with their exact types (netCDF 3 turned int64
into int32 and text into objects, so the content hash changed and every node downstream ran again), and nothing in
the folder can run code when it is read (the cache lives in the project, which may come from someone else; arrays
are read with allow_pickle=False). The content hashes are stored too, so a node that takes a cached output sees the
same input hash as after the original run, also where attributes come back as plain Python numbers. An output that
cannot be stored this way (an object array) is simply not cached.
"""
import json
import os
import shutil

import numpy as np
import xarray as xr

from tesselab.analysis import write_json


def folder(project_root, fingerprint) -> str:
    return os.path.join(project_root, "tesselab", "cache", fingerprint[:2], fingerprint)


def _plain(value):
    """Attributes as JSON can hold them (numpy numbers and arrays as Python ones)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} cannot be cached")


def load(project_root, fingerprint) -> dict | None:
    """{'outputs': ..., 'hashes': ..., 'diagnostics': ..., 'seconds': ...} or None if not cached."""
    path = folder(project_root, fingerprint)
    meta = os.path.join(path, "meta.json")
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
        outputs, hashes = {}, {}
        for name, entry in info["outputs"].items():
            with np.load(os.path.join(path, f"{name}.npz"), allow_pickle=False) as arrays:
                data_vars, coords = {}, {}
                for i, var in enumerate(entry["variables"]):
                    (coords if var["coord"] else data_vars)[var["name"]] = (var["dims"], arrays[f"v{i}"],
                                                                         var["attrs"])
            outputs[name] = xr.Dataset(data_vars, coords, entry["attrs"])
            if entry.get("hash"):
                hashes[name] = entry["hash"]
        return {"outputs": outputs, "hashes": hashes, "diagnostics": info.get("diagnostics", []),
                "seconds": info.get("seconds", 0.0), "error": None, "cached": "disk"}
    except Exception:  # a broken entry: computed again
        return None


def store(project_root, fingerprint, result, hashes=None) -> bool:
    """Keep a result (hashes: {output: content hash}); False if an output cannot be stored (it is then simply not
    cached)."""
    path = folder(project_root, fingerprint)
    temporary = path + f".tmp{os.getpid()}"
    try:
        os.makedirs(temporary, exist_ok=True)
        entries = {}
        for name, dataset in result["outputs"].items():
            variables, arrays = [], {}
            for i, (var_name, var) in enumerate(dataset.variables.items()):
                values = np.asarray(var.values)
                if values.dtype.hasobject:
                    raise TypeError(f"{name}.{var_name} holds Python objects")
                arrays[f"v{i}"] = values
                variables.append({"name": str(var_name), "coord": var_name in dataset.coords, "dims": list(var.dims),
                                  "attrs": json.loads(json.dumps(var.attrs, default=_plain))})
            with open(os.path.join(temporary, f"{name}.npz"), "wb") as f:
                np.savez(f, **arrays)
            entries[name] = {"variables": variables, "attrs": json.loads(json.dumps(dataset.attrs, default=_plain)),
                             "hash": (hashes or {}).get(name)}
        write_json(os.path.join(temporary, "meta.json"),
                   {"outputs": entries, "diagnostics": json.loads(json.dumps(result.get("diagnostics", []),
                                                                             default=_plain)),
                    "seconds": result.get("seconds", 0.0)})
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.replace(temporary, path)
        return True
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        return False


def clear(project_root):
    shutil.rmtree(os.path.join(project_root, "tesselab", "cache"), ignore_errors=True)
