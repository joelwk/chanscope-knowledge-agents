"""Small runtime shims to keep Replit runs stable.

Currently Replit's Nix images sometimes miss a dynamic loader path for
``libstdc++.so.6`` which pandas/numpy depend on. When that happens you see
errors like ``libstdc++.so.6: cannot open shared object file`` or a partially
initialised ``pandas`` module.  The helper below discovers the library (from
common locations or via ``g++ -print-file-name``), adds its directory to
``LD_LIBRARY_PATH`` for the running process, and preloads it so subsequent
imports succeed.
"""

from __future__ import annotations

import logging
import os
import subprocess
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def ensure_libstdcxx() -> bool:
    """Ensure libstdc++.so.6 is discoverable and loaded.

    Returns:
        bool: True when the library was found (and loaded), False otherwise.
    """

    # 1) Common locations inside Replit/Nix and Debian-based containers
    common_candidates = [
        Path("/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/lib64/libstdc++.so.6"),
        Path("/usr/lib64/libstdc++.so.6"),
    ]

    # 2) Ask the toolchain where it would place the library
    for compiler in ("g++", "c++", "gcc"):
        try:
            out = subprocess.check_output(
                [compiler, "-print-file-name=libstdc++.so.6"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if "/" in out:
                common_candidates.append(Path(out))
        except Exception:
            # Compiler not installed in the environment; skip quietly
            continue

    # 3) Use ctypes util finder as a last resort
    try:
        found = find_library("stdc++")
        if found and "/" in found:
            common_candidates.append(Path(found))
    except Exception:
        pass

    lib_path = _first_existing(common_candidates)
    if not lib_path:
        logger.warning(
            "libstdc++.so.6 not found. Install a libstdc++ runtime (e.g. add pkgs.stdenv.cc.cc.lib to replit.nix)"
        )
        return False

    lib_dir = str(lib_path.parent)
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in current_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current_ld}" if current_ld else lib_dir
        logger.info("Added %s to LD_LIBRARY_PATH", lib_dir)

    try:
        CDLL(str(lib_path))
        logger.debug("Preloaded libstdc++ from %s", lib_path)
        return True
    except Exception as exc:  # pragma: no cover - highly environment dependent
        logger.warning("Unable to preload libstdc++ (%s): %s", lib_path, exc)
        return False

