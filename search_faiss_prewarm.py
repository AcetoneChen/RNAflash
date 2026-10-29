"""
usage:
    python search_faiss_prewarm.py              # index + fasta
    python search_faiss_prewarm.py --index-only
    python search_faiss_prewarm.py --fasta-only
    python search_faiss_prewarm.py --check      # only report residency ratio, read no bytes
"""
import argparse
import ctypes
import os
from time import time

import numpy as np

# keep the same with search_faiss_mmp.py
INDEX = 'datasets/RNAcentral_27/faiss/faiss_index_fp32.index'
DATABASE_FASTA = 'datasets/RNAcentral_27/rnacentral_active_1022_v27.fasta'

SKIP_IF_RESIDENT = 0.98       # if residency ratio is above this, consider prewarming unnecessary
CHUNK = 8 << 20               # chunk size for sequential reads


# ---------------------------------------------------------------------------
# Residency ratio: one mincore pass, cost ≈ 0
# ---------------------------------------------------------------------------
PAGE = os.sysconf('SC_PAGE_SIZE')
_libc = ctypes.CDLL('libc.so.6', use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def resident_ratio(path):
    """What fraction of the file is already in page cache. Return 0 if unavailable (prefer reading one extra time)."""
    size = os.path.getsize(path)
    npages = (size + PAGE - 1) // PAGE
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, 1, 2, fd, 0)          # PROT_READ, MAP_PRIVATE
        if not addr or addr == ctypes.c_void_p(-1).value:
            return 0.0
        try:
            vec = ctypes.create_string_buffer(npages)
            if _libc.mincore(ctypes.c_void_p(addr),
                             ctypes.c_size_t(npages * PAGE), vec) != 0:
                return 0.0
            # 34.8GB = 8.5 million pages. Count bits with numpy, not a Python loop (which costs an extra 0.65s)
            bits = np.frombuffer(vec.raw[:npages], dtype=np.uint8)
            return float(np.count_nonzero(bits & 1)) / npages
        finally:
            _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Sequential read: the only way to actually pull the file into page cache
#
# Why not use posix_fadvise(POSIX_FADV_WILLNEED): measured on this machine it returns in 0.0ms, and
# after 30s the target range's residency ratio does not move at all; completely ineffective. MADV_WILLNEED is equally ineffective.
# Only actually reading once will put it into page cache.
# ---------------------------------------------------------------------------

def _read_through(path, label, chunk=CHUNK, verbose=True):
    """Read the file sequentially once, return (bytes read, residency ratio before reading, elapsed time). Bytes read of 0 means skipped."""
    t0 = time()
    frac = resident_ratio(path)
    if frac >= SKIP_IF_RESIDENT:
        if verbose:
            print(f'[prewarm] {label} already resident {frac * 100:.1f}%, '
                  f'skipped ({time() - t0:.2f}s)', flush=True)
        return 0, frac, time() - t0

    nread = 0
    with open(path, 'rb', buffering=0) as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            nread += len(buf)

    dt = time() - t0
    if verbose:
        print(f'[prewarm] {label} {nread / 2**30:.2f} GB in {dt:.1f}s '
              f'({nread / 2**20 / dt:.0f} MB/s), residency before read {frac * 100:.1f}%',
              flush=True)
    return nread, frac, dt


def prewarm(jobs, verbose=True):
    """jobs: [(path, label), ...] -- sequential reads serially. Return total elapsed time (seconds)."""
    t0 = time()
    for path, label in jobs:
        _read_through(path, label, verbose=verbose)
    return time() - t0




def main(do_index=True, do_fasta=True, check_only=False):
    jobs = []
    if do_index:
        jobs.append((INDEX, 'faiss index'))
    if do_fasta:
        jobs.append((DATABASE_FASTA, 'database fasta'))
    if not jobs:
        print('No data for prewarm.')
        return

    if check_only:
        t0 = time()
        for path, label in jobs:
            print(f'{label:16s} {resident_ratio(path) * 100:6.1f}% resident   '
                  f'({os.path.getsize(path) / 2**30:6.2f} GB)')
        print(f'[check] {time() - t0:.2f}s  (read no data)')
        return

    t_total = prewarm(jobs)
    print(f'[prewarm] total {t_total:.2f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index-only', action='store_true')
    ap.add_argument('--fasta-only', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    only = a.index_only or a.fasta_only
    main(do_index=a.index_only or not only,
         do_fasta=a.fasta_only or not only,
         check_only=a.check)
