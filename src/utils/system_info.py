"""
src/utils/system_info.py
=========================
Host and Jetson Orin Nano system information collector.

Collected subsystems
--------------------
- **OS**      : distribution, kernel, hostname, architecture, Python version.
- **CPU**     : model name, physical / logical core count, base & boost
                frequency, current per-core frequency, cache sizes.
- **RAM**     : total, available, used (physical + swap), via psutil or
                ``/proc/meminfo`` fallback.
- **GPU**     : device count, per-device name / UUID / VRAM / driver /
                compute capability; pynvml primary, torch.cuda secondary.
- **CUDA**    : runtime version, cuDNN version, NVCC version.
- **JetPack** : version string, L4T version, Jetson board model,
                tegra chip ID, module info — all from Jetson-specific
                sysfs / dpkg paths; silently absent on non-Jetson hardware.
- **PyTorch** : version, build features, CUDA compiled version.
- **Thermal** : Jetson thermal-zone temperatures (if present).
- **Clocks**  : Jetson GPU / CPU / EMC frequencies (if present).

All subsystems degrade gracefully: missing tools, absent sysfs paths, and
import errors produce empty sub-dicts rather than exceptions.

Usage
-----
>>> from utils.system_info import collect, save
>>> info = collect()
>>> save(info, "runs/exp01/system_info.json")
>>> print(info["gpu"][0]["name"])
'NVIDIA Orin'

Or from the command line::

    python -m src.utils.system_info --out runs/exp01/system_info.json
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports (all soft)
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil          # type: ignore
    _PSUTIL = True
except ImportError:
    _psutil = None                    # type: ignore
    _PSUTIL = False

try:
    import torch as _torch            # type: ignore
    _TORCH = True
except ImportError:
    _torch = None                     # type: ignore
    _TORCH = False

try:
    import pynvml as _nvml            # type: ignore
    _nvml.nvmlInit()
    _NVML = True
except Exception:                     # noqa: BLE001
    _nvml = None                      # type: ignore
    _NVML = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, *, timeout: int = 5) -> str:
    """Run a shell command and return stripped stdout; empty string on failure."""
    try:
        out = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL,
            timeout=timeout, text=True,
        )
        return out.strip()
    except Exception:               # noqa: BLE001
        return ""


def _read(path: str, *, default: str = "") -> str:
    """Read a sysfs / procfs file; return ``default`` on any error."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:               # noqa: BLE001
        return default


def _to_mb(value_bytes: int) -> float:
    return round(value_bytes / 1_048_576, 2)


def _to_mhz(value_hz: int) -> float:
    return round(value_hz / 1_000_000, 2)


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------

def _collect_os() -> dict:
    info: Dict[str, Any] = {
        "hostname":    platform.node(),
        "os":          platform.system(),
        "release":     platform.release(),
        "version":     platform.version(),
        "machine":     platform.machine(),
        "architecture": platform.architecture()[0],
        "python":      sys.version.split()[0],
        "python_full": sys.version,
    }

    # Linux distribution details.
    if platform.system() == "Linux":
        # /etc/os-release (standard)
        os_release_raw = _read("/etc/os-release")
        if os_release_raw:
            for line in os_release_raw.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    info[f"os_{k.lower()}"] = v.strip('"')

        # Fallback: lsb_release
        if "os_pretty_name" not in info:
            info["os_pretty_name"] = _run("lsb_release -ds")

    return info


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _collect_cpu() -> dict:
    info: Dict[str, Any] = {
        "model":          _cpu_model(),
        "physical_cores": None,
        "logical_cores":  None,
        "base_freq_mhz":  None,
        "max_freq_mhz":   None,
        "current_freq_mhz": None,
        "per_core_freq_mhz": [],
        "cache": {},
    }

    # Core counts.
    if _PSUTIL:
        info["physical_cores"] = _psutil.cpu_count(logical=False)
        info["logical_cores"]  = _psutil.cpu_count(logical=True)
        freq = _psutil.cpu_freq()
        if freq:
            info["base_freq_mhz"]    = round(freq.min,     2)
            info["max_freq_mhz"]     = round(freq.max,     2)
            info["current_freq_mhz"] = round(freq.current, 2)
        per = _psutil.cpu_freq(percpu=True)
        if per:
            info["per_core_freq_mhz"] = [round(f.current, 2) for f in per]
    else:
        try:
            import os as _os
            info["logical_cores"] = _os.cpu_count()
        except Exception:           # noqa: BLE001
            pass
        # Try /proc/cpuinfo for frequency on Linux.
        raw = _read("/proc/cpuinfo")
        if raw:
            for line in raw.splitlines():
                if line.lower().startswith("cpu mhz"):
                    try:
                        info["current_freq_mhz"] = float(line.split(":")[1].strip())
                    except ValueError:
                        pass
                    break

    # Cache sizes from /sys (Linux only).
    cache_base = Path("/sys/devices/system/cpu/cpu0/cache")
    if cache_base.exists():
        for idx_dir in sorted(cache_base.iterdir()):
            level = _read(str(idx_dir / "level"))
            ctype = _read(str(idx_dir / "type"))
            size  = _read(str(idx_dir / "size"))
            if level and ctype and size:
                key = f"L{level}_{ctype}"
                info["cache"][key] = size

    # Load average (Unix only).
    try:
        la = os.getloadavg()
        info["load_avg_1m"]  = round(la[0], 2)
        info["load_avg_5m"]  = round(la[1], 2)
        info["load_avg_15m"] = round(la[2], 2)
    except (AttributeError, OSError):
        pass

    return info


def _cpu_model() -> str:
    """Best-effort CPU model string."""
    # Linux: /proc/cpuinfo
    raw = _read("/proc/cpuinfo")
    for line in raw.splitlines():
        for key in ("model name", "Hardware", "Processor"):
            if line.lower().startswith(key.lower()) and ":" in line:
                return line.split(":", 1)[1].strip()

    # macOS / Windows fallbacks.
    if platform.system() == "Darwin":
        return _run("sysctl -n machdep.cpu.brand_string") or platform.processor()
    if platform.system() == "Windows":
        return _run(
            'wmic cpu get Name /format:list'
        ).replace("Name=", "").strip() or platform.processor()

    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _collect_ram() -> dict:
    info: Dict[str, Any] = {}

    if _PSUTIL:
        vm = _psutil.virtual_memory()
        sw = _psutil.swap_memory()
        info = {
            "total_mb":     _to_mb(vm.total),
            "available_mb": _to_mb(vm.available),
            "used_mb":      _to_mb(vm.used),
            "percent":      round(vm.percent, 1),
            "swap_total_mb": _to_mb(sw.total),
            "swap_used_mb":  _to_mb(sw.used),
            "swap_percent":  round(sw.percent, 1),
        }
    else:
        # /proc/meminfo fallback (Linux / Jetson).
        raw = _read("/proc/meminfo")
        mem: Dict[str, int] = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    mem[key] = int(parts[1]) * 1024   # kB → bytes
                except ValueError:
                    pass
        if mem:
            total     = mem.get("MemTotal",     0)
            available = mem.get("MemAvailable",  0)
            used      = total - available
            info = {
                "total_mb":     _to_mb(total),
                "available_mb": _to_mb(available),
                "used_mb":      _to_mb(used),
                "percent":      round(used / max(total, 1) * 100, 1),
                "swap_total_mb": _to_mb(mem.get("SwapTotal", 0)),
                "swap_used_mb":  _to_mb(
                    mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
                ),
            }

    # Jetson unified memory note.
    if _is_jetson():
        info["unified_memory"] = True
        info["note"] = (
            "Jetson uses unified LPDDR5X shared between CPU and GPU; "
            "GPU VRAM is carved from this total."
        )

    return info


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _collect_gpu() -> List[dict]:
    """Return a list of per-device GPU dicts (empty list if no GPU found)."""
    devices: List[dict] = []

    # ── pynvml (most complete) ────────────────────────────────────────────
    if _NVML:
        try:
            n = _nvml.nvmlDeviceGetCount()
            for i in range(n):
                h = _nvml.nvmlDeviceGetHandleByIndex(i)
                dev: Dict[str, Any] = {"index": i}

                # Name.
                try:
                    dev["name"] = _nvml.nvmlDeviceGetName(h)
                    if isinstance(dev["name"], bytes):
                        dev["name"] = dev["name"].decode()
                except Exception:   # noqa: BLE001
                    dev["name"] = "unknown"

                # UUID.
                try:
                    dev["uuid"] = _nvml.nvmlDeviceGetUUID(h)
                    if isinstance(dev["uuid"], bytes):
                        dev["uuid"] = dev["uuid"].decode()
                except Exception:   # noqa: BLE001
                    dev["uuid"] = ""

                # VRAM.
                try:
                    mem = _nvml.nvmlDeviceGetMemoryInfo(h)
                    dev["vram_total_mb"] = _to_mb(mem.total)
                    dev["vram_used_mb"]  = _to_mb(mem.used)
                    dev["vram_free_mb"]  = _to_mb(mem.free)
                except Exception:   # noqa: BLE001
                    pass

                # Driver / power / temperature.
                try:
                    dev["driver_version"] = _nvml.nvmlSystemGetDriverVersion()
                    if isinstance(dev["driver_version"], bytes):
                        dev["driver_version"] = dev["driver_version"].decode()
                except Exception:   # noqa: BLE001
                    pass
                try:
                    dev["temperature_c"] = _nvml.nvmlDeviceGetTemperature(
                        h, _nvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:   # noqa: BLE001
                    pass
                try:
                    dev["power_draw_w"] = round(
                        _nvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 2
                    )
                    dev["power_limit_w"] = round(
                        _nvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, 2
                    )
                except Exception:   # noqa: BLE001
                    pass

                # Compute capability.
                try:
                    major, minor = _nvml.nvmlDeviceGetCudaComputeCapability(h)
                    dev["compute_capability"] = f"{major}.{minor}"
                except Exception:   # noqa: BLE001
                    pass

                # Utilisation.
                try:
                    util = _nvml.nvmlDeviceGetUtilizationRates(h)
                    dev["gpu_util_pct"] = util.gpu
                    dev["mem_util_pct"] = util.memory
                except Exception:   # noqa: BLE001
                    pass

                # SM / memory clock.
                try:
                    dev["sm_clock_mhz"]  = _nvml.nvmlDeviceGetClockInfo(
                        h, _nvml.NVML_CLOCK_SM
                    )
                    dev["mem_clock_mhz"] = _nvml.nvmlDeviceGetClockInfo(
                        h, _nvml.NVML_CLOCK_MEM
                    )
                except Exception:   # noqa: BLE001
                    pass

                devices.append(dev)
        except Exception as exc:    # noqa: BLE001
            log.debug("pynvml GPU enumeration failed: %s", exc)

    # ── torch.cuda fallback ───────────────────────────────────────────────
    if not devices and _TORCH and _torch.cuda.is_available():
        for i in range(_torch.cuda.device_count()):
            prop = _torch.cuda.get_device_properties(i)
            devices.append({
                "index":             i,
                "name":              prop.name,
                "vram_total_mb":     _to_mb(prop.total_memory),
                "compute_capability": f"{prop.major}.{prop.minor}",
                "multi_processors":  prop.multi_processor_count,
                "source":            "torch.cuda",
            })

    # ── nvidia-smi last resort ────────────────────────────────────────────
    if not devices:
        smi = _run(
            "nvidia-smi "
            "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap "
            "--format=csv,noheader,nounits"
        )
        for line in smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                devices.append({
                    "index":          int(parts[0]) if parts[0].isdigit() else 0,
                    "name":           parts[1],
                    "uuid":           parts[2],
                    "vram_total_mb":  float(parts[3]) if parts[3] else None,
                    "driver_version": parts[4],
                    "compute_capability": parts[5] if len(parts) > 5 else None,
                    "source":         "nvidia-smi",
                })

    return devices


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

def _collect_cuda() -> dict:
    info: Dict[str, Any] = {}

    # Runtime version from torch.
    if _TORCH:
        info["torch_cuda_version"] = getattr(_torch.version, "cuda", None)
        info["cudnn_version"]      = (
            str(_torch.backends.cudnn.version())
            if _torch.backends.cudnn.is_available() else None
        )
        info["cuda_available"]     = _torch.cuda.is_available()

    # NVCC version.
    nvcc = _run("nvcc --version")
    if nvcc:
        m = re.search(r"release\s+([\d.]+)", nvcc)
        info["nvcc_version"] = m.group(1) if m else nvcc.splitlines()[-1]

    # CUDA toolkit via /usr/local/cuda.
    cuda_ver_file = _read("/usr/local/cuda/version.txt")
    if not cuda_ver_file:
        cuda_ver_file = _read("/usr/local/cuda/version.json")
    if cuda_ver_file:
        info["cuda_toolkit_file"] = cuda_ver_file.splitlines()[0]

    # CUDA_HOME.
    info["cuda_home"] = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))

    return info


# ---------------------------------------------------------------------------
# JetPack / Jetson
# ---------------------------------------------------------------------------

def _is_jetson() -> bool:
    """Return True if running on a Jetson device."""
    return (
        Path("/etc/nv_tegra_release").exists()
        or Path("/etc/nv_tegra_release.txt").exists()
        or bool(_read("/proc/device-tree/model"))
        or "tegra" in _read("/proc/cpuinfo").lower()
    )


def _collect_jetpack() -> dict:
    info: Dict[str, Any] = {"is_jetson": False}
    if not _is_jetson():
        return info

    info["is_jetson"] = True

    # ── Board model ───────────────────────────────────────────────────────
    model = _read("/proc/device-tree/model")
    if not model:
        model = _run("cat /proc/device-tree/model 2>/dev/null || true")
    info["board_model"] = model.strip("\x00") or "unknown"

    # ── L4T (Linux for Tegra) version ─────────────────────────────────────
    tegra = _read("/etc/nv_tegra_release")
    if not tegra:
        tegra = _read("/etc/nv_tegra_release.txt")
    if tegra:
        info["l4t_raw"] = tegra.splitlines()[0]
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", tegra)
        if m:
            info["l4t_version"] = f"R{m.group(1)}.{m.group(2)}"

    # ── JetPack version via dpkg ───────────────────────────────────────────
    dpkg_jp = _run("dpkg -l nvidia-jetpack 2>/dev/null | grep '^ii'")
    if dpkg_jp:
        parts = dpkg_jp.split()
        info["jetpack_version"] = parts[2] if len(parts) > 2 else dpkg_jp
    else:
        # Fallback: check apt-show-versions or nv_jetson_info.
        jp_info = _read("/etc/nv_jetson_info")
        if jp_info:
            info["jetpack_raw"] = jp_info
            m = re.search(r"jetpack\s*=\s*([\d.]+)", jp_info, re.IGNORECASE)
            if m:
                info["jetpack_version"] = m.group(1)

    # ── Tegra chip revision ───────────────────────────────────────────────
    chip_id = _read("/sys/module/tegra_fuse/parameters/tegra_chip_id")
    if chip_id:
        info["tegra_chip_id"] = chip_id

    fuse_id = _run("cat /sys/module/tegra_fuse/parameters/tegra_fuse_id 2>/dev/null")
    if fuse_id:
        info["tegra_fuse_id"] = fuse_id

    # ── Module info ───────────────────────────────────────────────────────
    module = _run(
        "cat /proc/device-tree/nvidia,dtsfilename 2>/dev/null | tr -d '\\0'"
    )
    if module:
        info["dts_filename"] = module

    # ── NV power mode ─────────────────────────────────────────────────────
    pmode = _run("nvpmodel -q 2>/dev/null | head -2")
    if pmode:
        info["nvpmodel"] = pmode

    # ── jetson_clocks status ──────────────────────────────────────────────
    jc_status = _run("jetson_clocks --show 2>/dev/null | head -5")
    if jc_status:
        info["jetson_clocks_status"] = jc_status

    return info


# ---------------------------------------------------------------------------
# Thermal zones (Jetson / Linux)
# ---------------------------------------------------------------------------

def _collect_thermals() -> Dict[str, float]:
    """Read all /sys/class/thermal/thermal_zone* temperatures in °C."""
    temps: Dict[str, float] = {}
    base = Path("/sys/class/thermal")
    if not base.exists():
        return temps
    for zone in sorted(base.iterdir()):
        type_file = zone / "type"
        temp_file = zone / "temp"
        if type_file.exists() and temp_file.exists():
            zone_type = _read(str(type_file)) or zone.name
            raw_temp  = _read(str(temp_file))
            try:
                temps[zone_type] = round(int(raw_temp) / 1000.0, 2)
            except ValueError:
                pass
    return temps


# ---------------------------------------------------------------------------
# Clock frequencies (Jetson / Linux)
# ---------------------------------------------------------------------------

def _collect_clocks() -> Dict[str, Any]:
    """Read CPU, GPU and EMC frequencies from sysfs devfreq / cpufreq."""
    clocks: Dict[str, Any] = {}

    # GPU devfreq.
    for pattern in (
        "/sys/devices/gpu.0/devfreq/*/cur_freq",
        "/sys/class/devfreq/gpu/cur_freq",
        "/sys/devices/platform/gpu.0/devfreq/gpu.0/cur_freq",
    ):
        import glob
        matches = glob.glob(pattern)
        for m in matches:
            raw = _read(m)
            if raw:
                try:
                    clocks["gpu_cur_hz"] = int(raw)
                    clocks["gpu_cur_mhz"] = _to_mhz(int(raw))
                except ValueError:
                    pass
                break

    # EMC (memory controller).
    for emc_path in (
        "/sys/kernel/nvpmodel_emc_cap/emc_iso_cap",
        "/sys/devices/platform/tegra-mc/la_r_a/cur_freq",
    ):
        raw = _read(emc_path)
        if raw:
            try:
                clocks["emc_cur_hz"] = int(raw)
                clocks["emc_cur_mhz"] = _to_mhz(int(raw))
            except ValueError:
                pass
            break

    # CPU per-core maximum frequency.
    cpu_freqs: List[int] = []
    import glob as _glob
    for freq_file in sorted(
        _glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
    ):
        raw = _read(freq_file)
        try:
            cpu_freqs.append(int(raw) * 1000)  # kHz → Hz
        except ValueError:
            pass
    if cpu_freqs:
        clocks["cpu_cur_hz_per_core"] = cpu_freqs
        clocks["cpu_cur_mhz_per_core"] = [_to_mhz(f) for f in cpu_freqs]
        clocks["cpu_max_mhz"] = _to_mhz(max(cpu_freqs))

    return clocks


# ---------------------------------------------------------------------------
# PyTorch build info
# ---------------------------------------------------------------------------

def _collect_pytorch() -> dict:
    if not _TORCH:
        return {"installed": False}
    info: Dict[str, Any] = {
        "installed":        True,
        "version":          _torch.__version__,
        "cuda_compiled":    getattr(_torch.version, "cuda", None),
        "debug_build":      _torch.version.debug,
        "git_version":      getattr(_torch.version, "git_version", None),
        "cuda_available":   _torch.cuda.is_available(),
        "cudnn_available":  _torch.backends.cudnn.is_available(),
        "cudnn_enabled":    _torch.backends.cudnn.enabled,
    }
    if _torch.cuda.is_available():
        info["cuda_device_count"] = _torch.cuda.device_count()
        info["current_device"]    = _torch.cuda.current_device()
        info["current_device_name"] = _torch.cuda.get_device_name(
            _torch.cuda.current_device()
        )
    # Build features (torch.backends).
    for backend in ("mkl", "mkldnn", "openmp", "nccl", "mps"):
        be = getattr(_torch.backends, backend, None)
        if be is not None:
            try:
                info[f"backend_{backend}"] = be.is_available()
            except Exception:           # noqa: BLE001
                pass
    return info


# ---------------------------------------------------------------------------
# Top-level collector
# ---------------------------------------------------------------------------

def collect(*, verbose: bool = False) -> Dict[str, Any]:
    """
    Collect all system information and return a nested dict.

    Parameters
    ----------
    verbose : If True, log each subsystem as it is collected.

    Returns
    -------
    dict with keys:
        collected_at, os, cpu, ram, gpu, cuda, jetpack,
        thermals, clocks, pytorch.

    Examples
    --------
    >>> info = collect()
    >>> info["jetpack"]["jetpack_version"]
    '5.1.2'
    >>> info["gpu"][0]["name"]
    'NVIDIA Orin'
    """
    result: Dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    subsystems = [
        ("os",      _collect_os),
        ("cpu",     _collect_cpu),
        ("ram",     _collect_ram),
        ("gpu",     _collect_gpu),
        ("cuda",    _collect_cuda),
        ("jetpack", _collect_jetpack),
        ("thermals", _collect_thermals),
        ("clocks",  _collect_clocks),
        ("pytorch", _collect_pytorch),
    ]

    for name, fn in subsystems:
        t0 = time.monotonic()
        try:
            result[name] = fn()
        except Exception as exc:        # noqa: BLE001
            log.warning("system_info: failed to collect '%s': %s", name, exc)
            result[name] = {"error": str(exc)}
        if verbose:
            log.debug(
                "system_info: collected '%s' in %.1f ms",
                name, (time.monotonic() - t0) * 1000,
            )

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(info: Dict[str, Any], path: "str | Path") -> Path:
    """
    Atomically write ``info`` as indented JSON to ``path``.

    The parent directory is created if it does not exist.

    Parameters
    ----------
    info : Dict returned by ``collect()``.
    path : Destination file path.

    Returns
    -------
    Path  — the written file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, default=str, ensure_ascii=False)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("System info saved to %s", out)
    return out


def load(path: "str | Path") -> Dict[str, Any]:
    """Load a previously saved system-info JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def format_summary(info: Dict[str, Any]) -> str:
    """
    Return a compact, human-readable multi-line summary string.

    Examples
    --------
    >>> print(format_summary(collect()))
    ── System Info ────────────────────────────────────
    Collected : 2025-05-11T14:23:01+00:00
    OS        : Ubuntu 20.04.6 LTS  (aarch64)  Linux 5.10.65
    CPU       : ARMv8 Processor  8 cores  @ 2015 MHz
    RAM       : 15812 MB total  |  11203 MB available  (29.2% used)
    GPU[0]    : NVIDIA Orin  7980 MB VRAM
    CUDA      : 11.4  cuDNN 8302
    JetPack   : 5.1.2  (L4T R35.3)
    PyTorch   : 2.1.0a0+41361538.nv23.6  (CUDA 11.4)
    ────────────────────────────────────────────────────
    """
    sep = "─" * 50
    lines = [f"── System Info {sep[14:]}"]
    lines.append(f"Collected : {info.get('collected_at', '?')}")

    # OS
    os_i = info.get("os", {})
    pretty = (
        os_i.get("os_pretty_name")
        or f"{os_i.get('os', '?')} {os_i.get('release', '')}".strip()
    )
    lines.append(
        f"OS        : {pretty}  ({os_i.get('machine', '?')})  "
        f"{os_i.get('os', '')} {os_i.get('release', '')}"
    )

    # CPU
    cpu_i = info.get("cpu", {})
    freq = cpu_i.get("current_freq_mhz") or cpu_i.get("max_freq_mhz") or "?"
    lines.append(
        f"CPU       : {cpu_i.get('model', '?')}  "
        f"{cpu_i.get('logical_cores', '?')} cores  @ {freq} MHz"
    )

    # RAM
    ram_i = info.get("ram", {})
    lines.append(
        f"RAM       : {ram_i.get('total_mb', '?')} MB total  |  "
        f"{ram_i.get('available_mb', '?')} MB available  "
        f"({ram_i.get('percent', '?')}% used)"
    )

    # GPU
    gpus = info.get("gpu", [])
    if isinstance(gpus, list):
        for g in gpus:
            vram = g.get("vram_total_mb", "?")
            lines.append(
                f"GPU[{g.get('index', '?')}]    : {g.get('name', '?')}  "
                f"{vram} MB VRAM"
                + (f"  CC {g['compute_capability']}" if "compute_capability" in g else "")
            )
    elif isinstance(gpus, dict) and "error" in gpus:
        lines.append(f"GPU       : unavailable ({gpus['error']})")
    else:
        lines.append("GPU       : none detected")

    # CUDA
    cuda_i = info.get("cuda", {})
    cv = cuda_i.get("torch_cuda_version") or cuda_i.get("nvcc_version") or "?"
    cdnn = cuda_i.get("cudnn_version", "?")
    lines.append(f"CUDA      : {cv}  cuDNN {cdnn}")

    # JetPack
    jp = info.get("jetpack", {})
    if jp.get("is_jetson"):
        jpv = jp.get("jetpack_version", "?")
        l4t = jp.get("l4t_version", "")
        board = jp.get("board_model", "")
        lines.append(f"JetPack   : {jpv}  ({l4t})  {board}")
    else:
        lines.append("JetPack   : not a Jetson device")

    # PyTorch
    pt = info.get("pytorch", {})
    if pt.get("installed"):
        ptv = pt.get("version", "?")
        ptcu = pt.get("cuda_compiled", "?")
        lines.append(f"PyTorch   : {ptv}  (CUDA {ptcu})")
    else:
        lines.append("PyTorch   : not installed")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and display host / Jetson system information."
    )
    parser.add_argument(
        "--out", "-o",
        metavar="PATH",
        help="Save system info to this JSON file.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    data = collect(verbose=args.verbose)
    print(format_summary(data))

    if args.out:
        saved = save(data, args.out)
        print(f"\nSaved to: {saved}")
