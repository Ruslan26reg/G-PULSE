"""Floating-point environment for exact reproduction.  Import this module BEFORE numpy (the scripts and tests do).

The paper's runs were computed with NumPy 2.5.3 / OpenBLAS 0.3.34 on an x86-64 CPU with AVX-512, one BLAS thread per
process.  Training amplifies last-bit differences, so the logged numbers are reproduced exactly only if the same
machine instructions are used:
  * OpenBLAS kernel 'SkylakeX' (AVX-512)                -> OPENBLAS_CORETYPE=SkylakeX
  * NumPy SIMD paths up to the AVX-512 level 'X86_V4'    -> NPY_ENABLE_CPU_FEATURES="X86_V3 X86_V4"
    (NumPy would otherwise also use newer AVX512_ICL / AVX512_SPR paths on newer CPUs, or AVX2 paths on older ones).
Both are set automatically on Linux machines whose CPU supports AVX-512 (F, CD, BW, DQ, VL), e.g. Intel Xeon
Skylake-SP and later, or AMD Zen 4 and later.  On other CPUs (AVX2-only x86, Apple Silicon, ...) nothing is set:
the results then agree with the paper statistically, not bit for bit.  Set GPULSE_EXACT=0 to disable this module;
variables already present in the environment are never overridden.
"""
import os

AVX512_FLAGS = ("avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl")


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return None                                         # not Linux: unknown


def configure():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")                   # multithreaded BLAS is not deterministic
    if os.environ.get("GPULSE_EXACT", "1") == "0":
        return False
    flags = cpu_flags()
    if not flags or not all(f in flags for f in AVX512_FLAGS):
        return False
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    if "NPY_DISABLE_CPU_FEATURES" not in os.environ:
        os.environ.setdefault("NPY_ENABLE_CPU_FEATURES", "X86_V3 X86_V4")
    return True


EXACT_MODE = configure()
