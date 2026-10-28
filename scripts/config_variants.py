"""Shared helpers for config/variants.yaml -- named overlays that each
only list the keys they change, merged onto config/quadrotor.yaml at run
time (same idea as Hydra config groups, scaled down to this repo's actual
needs: no framework, just a dict merge + a content hash).

A variant entry can touch anything in quadrotor.yaml, including
obstacle_avoidance.mode (see generate_mpc.py's OBS_MODE) which changes
the OCP's cost/constraint structure, not just a weight value -- when mode
differs from the base config, the overlay must also supply a
cost.W_stage/W_terminal pair already shaped for that mode (see
generate_mpc.py's shape assertions).
"""
import hashlib
import json
import os

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VARIANTS_PATH = os.path.join(REPO_ROOT, "config", "variants.yaml")
BASE_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "quadrotor.yaml")

# Files whose content affects what a solve actually computes -- included in
# every build/result hash alongside the resolved config, so a source edit
# (e.g. hand-tweaking generate_mpc.py) invalidates stale cached results
# instead of silently reusing them.
CODEGEN_SOURCES = [
    "scripts/generate_mpc.py",
    "src/argus_mpc/QuadrotorMpc.cpp",
    "include/argus_mpc/QuadrotorMpc.hpp",
    "examples/online_mapping_demo.cpp",
]


def deep_merge(base: dict, overlay: dict) -> dict:
    """overlay's keys win; nested dicts merge recursively instead of
    replacing the whole sub-tree (so an overlay touching only
    cost.W_stage doesn't have to also repeat every other cost.* key)."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_base_config() -> dict:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_all_variants() -> dict:
    if not os.path.exists(VARIANTS_PATH):
        return {}
    with open(VARIANTS_PATH) as f:
        return yaml.safe_load(f) or {}


def load_variant_overlay(name: str) -> dict:
    variants = load_all_variants()
    if name not in variants:
        raise KeyError(
            f"No such variant {name!r} in {VARIANTS_PATH}. Available: {sorted(variants)}")
    return variants[name] or {}


def resolve_variant(name: str | None) -> dict:
    """name=None (or 'baseline') -- the committed config as-is, no overlay."""
    base = load_base_config()
    if name is None or name == "baseline":
        return base
    return deep_merge(base, load_variant_overlay(name))


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build_hash(resolved_cfg: dict, extra_paths=CODEGEN_SOURCES) -> str:
    """Identifies 'this exact solver behavior': the resolved config
    (order-independent -- sort_keys) plus the content of every source
    file that can change what a solve computes. Two different variant
    names that happen to resolve to the same config + same source get
    the same hash on purpose (one cache entry, not a wasted rebuild)."""
    parts = [json.dumps(resolved_cfg, sort_keys=True)]
    for rel in extra_paths:
        path = os.path.join(REPO_ROOT, rel)
        with open(path) as f:
            parts.append(f.read())
    return content_hash(*parts)[:16]
