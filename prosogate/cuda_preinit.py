"""Pre-initialize CUDA driver context and fix LD_LIBRARY_PATH for driver 535.

This module solves three problems specific to this host:

1. **CUDA compat library hides real driver**.
   `/usr/local/cuda-12.2/compat/libcuda.so.535.104.05` resolves before the
   system `/lib64/libcuda.so.535.154.05`. The compat lib caps usable CUDA at
   12.2 — anything cu12.4+ (including torch 2.6+/cu124, 2.9+/cu126) triggers
   error 803 on first `cudaGetDeviceCount`.

2. **System cuDNN 9.2.0 overrides torch's bundled cuDNN 9.10.2**.
   `/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn.so.9` is older than
   what torch 2.9.x expects. pyannote LSTM forward fails with "cuDNN version
   incompatibility".

3. **Torch's cudart lazy init mishandles error 803**.
   Even after fixing #1, the first `cudaGetDeviceCount` from torch can still
   return 803. Calling `libcuda.cuInit(0)` from `libcuda.so` first warms up
   the driver API so cudart succeeds.

We do all three at import time, BEFORE `import torch`:
- Strip `*compat*` and `cuda-12.2/targets` paths from LD_LIBRARY_PATH.
- Prepend the conda nvidia/cudnn/lib path so torch finds its bundled cuDNN.
- Call libcuda.cuInit(0) via ctypes.

Note: LD_LIBRARY_PATH changes only take effect for SUBPROCESSES — they don't
affect the current process's dynamic linker. So if this module is imported
AFTER torch already loaded a wrong libcuda, it can't undo that. Always import
`prosogate.cuda_preinit` as the FIRST prosogate import, before importing torch
or pyannote.

If you need to spawn child processes (rare), re-export LD_LIBRARY_PATH there.
"""

from __future__ import annotations

import ctypes
import os
import sys

_INITIALIZED = False


def _fixed_ld_library_path() -> str:
    """Drop cuda compat / cuda-12.2 targets paths; prepend conda cuDNN."""
    parts = (os.environ.get("LD_LIBRARY_PATH") or "").split(":")
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if "compat" in p:
            continue
        if "cuda-12.2/targets" in p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    # Prepend conda's bundled cuDNN so torch finds 9.10.x not system 9.2.x
    cudnn_dir = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                             "site-packages", "nvidia", "cudnn", "lib")
    if os.path.isdir(cudnn_dir) and cudnn_dir not in seen:
        out.insert(0, cudnn_dir)

    return ":".join(out)


def init() -> bool:
    """Apply fixes. Returns True if all three steps succeed.

    Safe to call multiple times. If CUDA_VISIBLE_DEVICES is set to empty
    string the user explicitly disabled GPU and we skip the cuInit step.

    If LD_LIBRARY_PATH needs fixing AND torch hasn't been imported yet, we
    re-exec the current python process with the fixed LD_LIBRARY_PATH so the
    dynamic linker picks up the right libcuda / cuDNN. The re-exec is gated
    by an env var to avoid infinite loops.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    fixed = _fixed_ld_library_path()
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    needs_re_exec = (fixed != cur_ld)

    if needs_re_exec and not os.environ.get("PROSOGATE_CUDA_PREINIT_DONE"):
        # Only re-exec if torch hasn't been imported yet (otherwise LD change
        # is too late). Re-exec replaces this process with python under a
        # fixed LD_LIBRARY_PATH.
        if "torch" not in sys.modules:
            new_env = dict(os.environ)
            new_env["LD_LIBRARY_PATH"] = fixed
            new_env["PROSOGATE_CUDA_PREINIT_DONE"] = "1"
            try:
                os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)
            except OSError:
                pass  # fall through to in-process best effort

    # In-process best effort (only useful for child processes we spawn)
    if needs_re_exec:
        os.environ["LD_LIBRARY_PATH"] = fixed

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd == "":
        return False

    for path in ("/lib64/libcuda.so.1",
                 "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                 "libcuda.so.1"):
        try:
            libcuda = ctypes.CDLL(path)
        except OSError:
            continue
        try:
            ret = libcuda.cuInit(0)
            if ret == 0:
                _INITIALIZED = True
                return True
        except Exception:
            continue
    return False


# Auto-init on module import (must happen before `import torch`)
init()

