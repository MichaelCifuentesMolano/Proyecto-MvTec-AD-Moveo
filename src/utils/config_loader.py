"""
src/utils/config_loader.py
===========================
YAML experiment configuration loader for the MVTec-AD pipeline.

Features
--------
- Load one or more YAML files with ``load_config()``.
- **Inheritance** : any config may declare ``base: path/to/base.yaml``; the
  loader deep-merges child values on top of the base (arbitrarily deep).
- **Variable interpolation** : ``${VAR}`` tokens are resolved from
  environment variables or from a ``vars:`` section in the same file.
- **Environment overrides** : keys can be overridden at launch time via
  ``MVTEC_CFG__section__key=value`` environment variables (double-underscore
  as path separator, values auto-cast to int / float / bool / str).
- **Validation** : an optional ``schema:`` mapping (JSON-Schema-compatible
  subset) is enforced after loading.
- **Schema sections** recognised by the pipeline:

    experiment:   name, seed, output_dir, tags
    data:         dataset_root, categories, image_size, batch_size, num_workers
    model:        architecture, backbone, pretrained
    quantization: enabled, bits_weights, bits_activations, qat_epochs
    nas:          algorithm, n_generations, population_size, objectives,
                  crossover_prob, mutation_prob, reference_point
    training:     epochs, lr, weight_decay, scheduler, warmup_epochs
    deployment:   target_device, precision, trt_workspace_gb, onnx_opset
    tracking:     algorithm, min_hits, max_misses, iou_threshold,
                  alarm_threshold, frame_width, frame_height
    logging:      log_dir, level, csv, jsonl

Usage
-----
>>> cfg = load_config("configs/experiment.yaml")
>>> cfg.experiment.name
'mvtec_nsga2_int8'
>>> cfg.nas.n_generations
50

>>> # Multi-file merge (right wins on conflict):
>>> cfg = load_config(["configs/base.yaml", "configs/jetson.yaml"])

>>> # CLI-style overrides:
>>> cfg = load_config("configs/exp.yaml",
...                   overrides={"nas.n_generations": 100,
...                              "training.lr": 1e-4})
"""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# YAML is the only mandatory external dependency.
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_Scalar = Union[str, int, float, bool, None]
_Tree   = Dict[str, Any]

# Environment-variable prefix for run-time overrides.
_ENV_PREFIX = "MVTEC_CFG__"

# Interpolation pattern: ${VAR_NAME}
_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# ConfigNode — attribute-access dict
# ---------------------------------------------------------------------------

class ConfigNode:
    """
    Recursive dict wrapper that exposes keys as attributes.

    Supports both attribute access and dict-style access:

    >>> node = ConfigNode({"nas": {"n_generations": 50}})
    >>> node.nas.n_generations
    50
    >>> node["nas"]["n_generations"]
    50
    >>> "nas" in node
    True

    Mutating an existing key is allowed; adding new keys at runtime is
    intentionally disallowed to prevent typo-driven silent misconfiguration.
    """

    def __init__(self, data: _Tree) -> None:
        object.__setattr__(self, "_data", {})
        for k, v in data.items():
            object.__getattribute__(self, "_data")[k] = (
                ConfigNode(v) if isinstance(v, dict) else v
            )

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(
            f"Config has no key '{name}'. "
            f"Available: {sorted(data.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(
                f"Cannot add new config key '{name}' at runtime. "
                f"Define it in the YAML file."
            )
        data[name] = ConfigNode(value) if isinstance(value, dict) else value

    # ------------------------------------------------------------------
    # Dict-style access and iteration
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_data"))

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        return data.get(key, default)

    def keys(self):
        return object.__getattribute__(self, "_data").keys()

    def values(self):
        return object.__getattribute__(self, "_data").values()

    def items(self):
        return object.__getattribute__(self, "_data").items()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> _Tree:
        """Recursively convert back to a plain ``dict``."""
        out: _Tree = {}
        for k, v in object.__getattribute__(self, "_data").items():
            out[k] = v.to_dict() if isinstance(v, ConfigNode) else v
        return out

    def to_yaml(self) -> str:
        """Serialise to a YAML string (requires PyYAML)."""
        _require_yaml()
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        inner = ", ".join(f"{k}={v!r}" for k, v in list(data.items())[:6])
        suffix = ", …" if len(data) > 6 else ""
        return f"ConfigNode({{{inner}{suffix}}})"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: _Tree, override: _Tree) -> _Tree:
    """
    Recursively merge ``override`` into a copy of ``base``.

    - Dicts are merged recursively (override wins on conflict).
    - Lists and scalars in ``override`` replace ``base`` entirely.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# ---------------------------------------------------------------------------
# Variable interpolation
# ---------------------------------------------------------------------------

def _collect_vars(tree: _Tree) -> Dict[str, str]:
    """Extract the top-level ``vars:`` section as a string→string dict."""
    raw = tree.get("vars", {})
    return {k: str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _interpolate(value: Any, var_map: Dict[str, str]) -> Any:
    """
    Replace ``${VAR}`` tokens in string values (depth-first).

    Resolution order:
    1. ``var_map`` (from ``vars:`` section + env overrides fed in by caller).
    2. ``os.environ``.

    Raises ``KeyError`` if the variable is not found in either source.
    """
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            name = m.group(1)
            if name in var_map:
                return var_map[name]
            env_val = os.environ.get(name)
            if env_val is not None:
                return env_val
            raise KeyError(
                f"Interpolation variable '${{{name}}}' not found in "
                f"vars section or environment."
            )
        return _INTERP_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, var_map) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(item, var_map) for item in value]
    return value


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------

def _cast_env_value(raw: str) -> _Scalar:
    """
    Auto-cast an environment-variable string to int / float / bool / str.

    Conversion order: bool literals → int → float → str.
    """
    if raw.lower() in ("true", "yes", "1"):
        return True
    if raw.lower() in ("false", "no", "0"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _collect_env_overrides() -> Dict[str, _Scalar]:
    """
    Scan ``os.environ`` for ``MVTEC_CFG__*`` variables and return a flat dict
    mapping dot-separated key paths to cast values.

    Example::

        MVTEC_CFG__nas__n_generations=100
        → {"nas.n_generations": 100}
    """
    overrides: Dict[str, _Scalar] = {}
    prefix_len = len(_ENV_PREFIX)
    for env_key, env_val in os.environ.items():
        if env_key.startswith(_ENV_PREFIX):
            path = env_key[prefix_len:].replace("__", ".")
            overrides[path] = _cast_env_value(env_val)
    return overrides


def _apply_flat_overrides(tree: _Tree, overrides: Dict[str, Any]) -> _Tree:
    """
    Apply dot-separated key-path overrides to ``tree`` (mutates in-place copy).

    ``overrides`` keys may be:
    - Dot-separated paths : ``"nas.n_generations"`` → ``tree["nas"]["n_generations"]``
    - Top-level keys      : ``"seed"`` → ``tree["seed"]``

    Missing intermediate dicts are created automatically.
    """
    result = copy.deepcopy(tree)
    for path, value in overrides.items():
        parts = path.split(".")
        node = result
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        leaf = parts[-1]
        if leaf in node and isinstance(node[leaf], dict) and isinstance(value, dict):
            node[leaf] = _deep_merge(node[leaf], value)
        else:
            node[leaf] = value
    return result


# ---------------------------------------------------------------------------
# YAML loading with inheritance
# ---------------------------------------------------------------------------

def _require_yaml() -> None:
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required.  Install it with: pip install pyyaml"
        )


def _load_yaml_file(path: Path) -> _Tree:
    """Load a single YAML file.  Raises ``FileNotFoundError`` if absent."""
    _require_yaml()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must be a YAML mapping at the top level: {path}"
        )
    return data


def _resolve_with_inheritance(path: Path, _visited: Optional[set] = None) -> _Tree:
    """
    Load ``path`` and recursively resolve any ``base:`` directives.

    Inheritance is depth-first: the deepest ancestor is loaded first, then
    each child is merged on top.  Circular references raise ``ValueError``.
    """
    if _visited is None:
        _visited = set()
    resolved = path.resolve()
    if resolved in _visited:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    _visited.add(resolved)

    data = _load_yaml_file(path)
    base_rel = data.pop("base", None)

    if base_rel is not None:
        # Resolve relative to the directory of the current file.
        base_path = (path.parent / base_rel).resolve()
        log.debug("Config '%s' inherits from '%s'.", path.name, base_path.name)
        base_data = _resolve_with_inheritance(base_path, _visited)
        data = _deep_merge(base_data, data)

    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_schema(tree: _Tree, schema: _Tree, path: str = "") -> List[str]:
    """
    Lightweight JSON-Schema-compatible subset validation.

    Supported keywords per field:  ``type``, ``required``, ``minimum``,
    ``maximum``, ``enum``, ``properties`` (nested).

    Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []

    _TYPE_MAP = {
        "string":  str,
        "integer": int,
        "number":  (int, float),
        "boolean": bool,
        "array":   list,
        "object":  dict,
        "null":    type(None),
    }

    props = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in tree:
            errors.append(f"{path + key!r} is required but missing.")

    for key, field_schema in props.items():
        full_key = f"{path}{key}."
        if key not in tree:
            continue
        val = tree[key]

        # type check
        type_name = field_schema.get("type")
        if type_name and type_name in _TYPE_MAP:
            expected = _TYPE_MAP[type_name]
            if not isinstance(val, expected):
                errors.append(
                    f"{full_key.rstrip('.')} must be of type {type_name!r}, "
                    f"got {type(val).__name__!r}."
                )

        # enum
        if "enum" in field_schema and val not in field_schema["enum"]:
            errors.append(
                f"{full_key.rstrip('.')} must be one of {field_schema['enum']!r}, "
                f"got {val!r}."
            )

        # numeric bounds
        if isinstance(val, (int, float)):
            if "minimum" in field_schema and val < field_schema["minimum"]:
                errors.append(
                    f"{full_key.rstrip('.')} minimum is {field_schema['minimum']}, "
                    f"got {val}."
                )
            if "maximum" in field_schema and val > field_schema["maximum"]:
                errors.append(
                    f"{full_key.rstrip('.')} maximum is {field_schema['maximum']}, "
                    f"got {val}."
                )

        # nested object
        if isinstance(val, dict) and "properties" in field_schema:
            errors.extend(_validate_schema(val, field_schema, full_key))

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    paths:     Union[str, Path, Sequence[Union[str, Path]]],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    env_overrides: bool = True,
    interpolate:   bool = True,
    validate:      bool = True,
    strict:        bool = False,
) -> ConfigNode:
    """
    Load one or more YAML config files and return a ``ConfigNode``.

    Parameters
    ----------
    paths : str | Path | list[str | Path]
        A single YAML path or an ordered list.  When multiple files are given,
        they are deep-merged left-to-right (rightmost wins on conflict).
    overrides : dict, optional
        Dot-separated key-path overrides applied after all file loading,
        e.g. ``{"nas.n_generations": 100, "training.lr": 3e-4}``.
    env_overrides : bool
        If True, scan environment variables with prefix ``MVTEC_CFG__`` and
        apply them as overrides (after ``overrides`` dict).
    interpolate : bool
        If True, resolve ``${VAR}`` tokens using the ``vars:`` section and
        ``os.environ``.
    validate : bool
        If True and a ``schema:`` section is present, validate the config
        against it.
    strict : bool
        If True, raise ``ConfigValidationError`` on any schema violation.
        If False, log warnings instead.

    Returns
    -------
    ConfigNode  — attribute-access config tree.

    Raises
    ------
    FileNotFoundError      : A specified YAML file does not exist.
    ConfigValidationError  : Schema violation when ``strict=True``.
    ImportError            : PyYAML is not installed.

    Examples
    --------
    >>> cfg = load_config("configs/experiment.yaml")
    >>> cfg.nas.n_generations
    50

    >>> cfg = load_config(
    ...     ["configs/base.yaml", "configs/jetson_int8.yaml"],
    ...     overrides={"experiment.name": "quick_test", "nas.n_generations": 5},
    ... )
    """
    # ── Normalise path list ───────────────────────────────────────────────
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]

    # ── Load and merge files ──────────────────────────────────────────────
    merged: _Tree = {}
    for p in path_list:
        log.debug("Loading config: %s", p)
        tree = _resolve_with_inheritance(p)
        merged = _deep_merge(merged, tree)

    # ── Collect variable map (vars: section + current env) ────────────────
    var_map = _collect_vars(merged)

    # ── Apply inline overrides ────────────────────────────────────────────
    if overrides:
        merged = _apply_flat_overrides(merged, overrides)

    # ── Apply environment overrides ───────────────────────────────────────
    if env_overrides:
        env_ovr = _collect_env_overrides()
        if env_ovr:
            log.info("Applying %d environment override(s): %s", len(env_ovr),
                     list(env_ovr.keys()))
            merged = _apply_flat_overrides(merged, env_ovr)

    # ── Interpolate variables ─────────────────────────────────────────────
    if interpolate:
        # Add any env vars that appeared as ${} tokens into the map.
        var_map.update({k: str(v) for k, v in merged.get("vars", {}).items()})
        try:
            merged = _interpolate(merged, var_map)
        except KeyError as exc:
            raise ConfigError(f"Variable interpolation failed: {exc}") from exc

    # ── Validate ──────────────────────────────────────────────────────────
    schema = merged.pop("schema", None)
    if validate and schema and isinstance(schema, dict):
        errors = _validate_schema(merged, schema)
        if errors:
            msg = "Config validation errors:\n" + "\n".join(f"  • {e}" for e in errors)
            if strict:
                raise ConfigValidationError(msg)
            for e in errors:
                log.warning("[config] %s", e)

    # Remove internal ``vars:`` section from the final node.
    merged.pop("vars", None)

    node = ConfigNode(merged)
    log.info(
        "Config loaded from %s file(s). Top-level keys: %s",
        len(path_list),
        sorted(node.keys()),
    )
    return node


def load_config_dict(
    data: _Tree,
    *,
    overrides:     Optional[Dict[str, Any]] = None,
    env_overrides: bool = False,
    interpolate:   bool = False,
) -> ConfigNode:
    """
    Build a ``ConfigNode`` directly from a Python dict (no file I/O).

    Useful for programmatic config construction in tests or notebooks.

    Parameters
    ----------
    data          : Plain dict representing the config tree.
    overrides     : Dot-path overrides applied on top.
    env_overrides : Apply ``MVTEC_CFG__*`` environment overrides.
    interpolate   : Resolve ``${VAR}`` tokens.

    Returns
    -------
    ConfigNode
    """
    merged = copy.deepcopy(data)
    if overrides:
        merged = _apply_flat_overrides(merged, overrides)
    if env_overrides:
        env_ovr = _collect_env_overrides()
        if env_ovr:
            merged = _apply_flat_overrides(merged, env_ovr)
    if interpolate:
        var_map = _collect_vars(merged)
        merged = _interpolate(merged, var_map)
    return ConfigNode(merged)


def save_config(node: ConfigNode, path: Union[str, Path]) -> None:
    """
    Serialise a ``ConfigNode`` back to YAML and write atomically.

    Parameters
    ----------
    node : Config to serialise.
    path : Destination file path (parent directories created if absent).
    """
    _require_yaml()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.dump(
                node.to_dict(),
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("Config saved to %s", out)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised for any config loading / interpolation error."""


class ConfigValidationError(ConfigError):
    """Raised when ``strict=True`` and the config fails schema validation."""


# ---------------------------------------------------------------------------
# Default schema for the pipeline
# ---------------------------------------------------------------------------

PIPELINE_SCHEMA: _Tree = {
    "properties": {
        "experiment": {
            "properties": {
                "name":       {"type": "string"},
                "seed":       {"type": "integer", "minimum": 0},
                "output_dir": {"type": "string"},
            },
            "required": ["name", "seed", "output_dir"],
        },
        "data": {
            "properties": {
                "dataset_root": {"type": "string"},
                "categories":   {"type": "array"},
                "image_size":   {"type": "integer", "minimum": 32, "maximum": 2048},
                "batch_size":   {"type": "integer", "minimum": 1},
                "num_workers":  {"type": "integer", "minimum": 0},
            },
            "required": ["dataset_root", "categories"],
        },
        "nas": {
            "properties": {
                "algorithm":      {"type": "string", "enum": ["nsga2", "nsga3", "random"]},
                "n_generations":  {"type": "integer", "minimum": 1},
                "population_size":{"type": "integer", "minimum": 4},
                "objectives":     {"type": "array"},
                "crossover_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "mutation_prob":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
        "quantization": {
            "properties": {
                "enabled":           {"type": "boolean"},
                "bits_weights":      {"type": "integer", "enum": [4, 8, 16, 32]},
                "bits_activations":  {"type": "integer", "enum": [4, 8, 16, 32]},
                "qat_epochs":        {"type": "integer", "minimum": 0},
            },
        },
        "training": {
            "properties": {
                "epochs":        {"type": "integer", "minimum": 1},
                "lr":            {"type": "number", "minimum": 0.0},
                "weight_decay":  {"type": "number", "minimum": 0.0},
                "warmup_epochs": {"type": "integer", "minimum": 0},
            },
        },
        "deployment": {
            "properties": {
                "target_device":    {"type": "string"},
                "precision":        {"type": "string", "enum": ["fp32", "fp16", "int8"]},
                "trt_workspace_gb": {"type": "number", "minimum": 0.0},
                "onnx_opset":       {"type": "integer", "minimum": 9, "maximum": 20},
            },
        },
        "tracking": {
            "properties": {
                "algorithm":       {"type": "string",
                                    "enum": ["sort", "lightweight", "csrt", "kcf"]},
                "min_hits":        {"type": "integer", "minimum": 1},
                "max_misses":      {"type": "integer", "minimum": 1},
                "iou_threshold":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "alarm_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "frame_width":     {"type": "integer", "minimum": 16},
                "frame_height":    {"type": "integer", "minimum": 16},
            },
        },
        "logging": {
            "properties": {
                "log_dir": {"type": "string"},
                "level":   {"type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "csv":     {"type": "boolean"},
                "jsonl":   {"type": "boolean"},
            },
        },
    },
}


def validate_pipeline_config(node: ConfigNode, *, strict: bool = True) -> List[str]:
    """
    Validate a loaded ``ConfigNode`` against the built-in pipeline schema.

    Parameters
    ----------
    node   : Config to validate.
    strict : If True, raise ``ConfigValidationError`` on any error.

    Returns
    -------
    List[str]  — error strings (empty = valid).
    """
    errors = _validate_schema(node.to_dict(), PIPELINE_SCHEMA)
    if errors and strict:
        msg = "Pipeline config validation failed:\n" + "\n".join(
            f"  • {e}" for e in errors
        )
        raise ConfigValidationError(msg)
    for e in errors:
        log.warning("[validate_pipeline_config] %s", e)
    return errors


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse ``key=value`` strings from a CLI argument list into a flat override dict.

    Values are auto-cast (bool → int → float → str).

    Parameters
    ----------
    args : List of ``"section.key=value"`` strings.

    Returns
    -------
    dict  — suitable for passing as ``overrides`` to ``load_config()``.

    Examples
    --------
    >>> _parse_cli_overrides(["nas.n_generations=20", "training.lr=1e-4"])
    {"nas.n_generations": 20, "training.lr": 0.0001}
    """
    result: Dict[str, Any] = {}
    for token in args:
        if "=" not in token:
            raise ConfigError(
                f"CLI override must be in 'key=value' format, got: {token!r}"
            )
        key, _, raw_val = token.partition("=")
        result[key.strip()] = _cast_env_value(raw_val.strip())
    return result


def load_config_from_cli(
    default_config: Union[str, Path],
    cli_args:       Optional[List[str]] = None,
    *,
    env_overrides:  bool = True,
    strict:         bool = False,
) -> ConfigNode:
    """
    Convenience loader for scripts: load ``default_config``, then apply any
    ``key=value`` overrides from ``cli_args`` (or ``sys.argv[1:]``).

    Parameters
    ----------
    default_config : Path to the default YAML config file.
    cli_args       : List of ``"section.key=value"`` strings.  If ``None``,
                     reads from ``sys.argv[1:]``.
    env_overrides  : Apply ``MVTEC_CFG__*`` environment overrides.
    strict         : Raise on schema validation errors.

    Returns
    -------
    ConfigNode

    Examples
    --------
    Script invocation::

        python train.py nas.n_generations=20 training.lr=1e-4

    Inside the script::

        cfg = load_config_from_cli("configs/experiment.yaml")
        cfg.nas.n_generations  # → 20
    """
    import sys as _sys
    if cli_args is None:
        # Filter out the script name and anything starting with '-' (flags).
        cli_args = [a for a in _sys.argv[1:] if "=" in a]

    overrides = _parse_cli_overrides(cli_args) if cli_args else {}
    return load_config(
        default_config,
        overrides=overrides,
        env_overrides=env_overrides,
        strict=strict,
    )
