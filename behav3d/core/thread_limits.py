"""OpenMP thread cap for scikit-learn KMeans (also run inside every GaussianHMM fit).

KMeans starts one OpenMP thread per logical CPU and every thread calls OpenBLAS
at the same time. The OpenBLAS bundled with numpy/scipy supports at most 128
concurrent callers; above that the process logs "OpenBLAS warning: precompiled
NUM_THREADS exceeded" and segfaults (seen on >128-CPU Linux workstations).
Capping at 64 threads avoids it on every OS without the user setting
OMP_NUM_THREADS. Results are identical; only the thread count changes.
"""
import os

from threadpoolctl import threadpool_info, threadpool_limits

MAX_OPENMP_THREADS = 64  # value tested and confirmed in the crash report


def openmp_thread_cap():
    """Context manager capping OpenMP threads at ``MAX_OPENMP_THREADS``.

    Never raises the count above its current value, so a lower ``OMP_NUM_THREADS``
    set by the user still wins (``threadpool_limits`` sets, it does not only lower).
    On machines with <= 64 CPUs nothing changes.
    """
    current = [info["num_threads"] for info in threadpool_info() if info.get("user_api") == "openmp"]
    return threadpool_limits(
        limits=min([MAX_OPENMP_THREADS, os.cpu_count() or 1, *current]),
        user_api="openmp",
    )
