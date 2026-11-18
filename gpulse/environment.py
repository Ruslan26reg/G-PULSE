"""Description of the hardware and software environment (stored in every run log and printed by env_report.py)."""
import os
import platform
import sys


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _mem_gb():
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        return None


def describe():
    import numpy as np
    import scipy
    env = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
               os=platform.platform(), machine=platform.machine(), cpu=_cpu_model(), logical_cpus=os.cpu_count(),
               memory_gb=_mem_gb(), gpu="none (all computations on the CPU)")
    try:
        from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__, __cpu_features__
        env["numpy_simd_baseline"] = list(__cpu_baseline__)
        env["numpy_simd_dispatch_in_use"] = [k for k, v in __cpu_features__.items() if v and k in __cpu_dispatch__]
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_info
        blas = [i for i in threadpool_info() if i.get("user_api") == "blas"]
        if blas:
            env["blas"] = dict(library=blas[0].get("internal_api"), version=blas[0].get("version"),
                               kernel=blas[0].get("architecture"), threads=blas[0].get("num_threads"))
    except Exception:                                   # threadpoolctl is optional
        pass
    for k in ("OPENBLAS_CORETYPE", "NPY_ENABLE_CPU_FEATURES", "NPY_DISABLE_CPU_FEATURES", "OMP_NUM_THREADS"):
        if k in os.environ:
            env.setdefault("variables", {})[k] = os.environ[k]
    return env


def exact_reproduction_possible(env):
    """True if this environment uses the same instructions as the paper's runs (see repro_env.py)."""
    ok_versions = env["numpy"] == "2.5.3" and env["scipy"] == "1.18.1"
    simd = env.get("numpy_simd_dispatch_in_use")
    ok_simd = simd is not None and "X86_V4" in simd and not any(s in simd for s in ("AVX512_ICL", "AVX512_SPR"))
    kernel = env.get("blas", {}).get("kernel")
    ok_blas = kernel == "SkylakeX" or (kernel is None and env.get("variables", {}).get("OPENBLAS_CORETYPE") == "SkylakeX")
    return dict(numpy_scipy_versions=ok_versions, numpy_simd_level=ok_simd, blas_kernel=ok_blas,
                exact=ok_versions and ok_simd and ok_blas)
