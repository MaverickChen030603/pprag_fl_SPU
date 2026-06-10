#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys

REQUIRED = {
    "transformers": "4.x",
    "torch": "2.x",
    "datasets": "2.x",
    "scipy": "1.x",
    "numpy": "1.x",
    "pandas": "1.x",
    "tqdm": "4.x",
}


def main() -> int:
    missing = []
    for pkg, _ver in REQUIRED.items():
        mod = pkg.replace("-", "_")
        try:
            m = importlib.import_module(mod)
            print(f"OK {pkg} {getattr(m, '__version__', 'unknown')}")
        except ImportError:
            print(f"MISSING {pkg} - pip install {pkg}")
            missing.append(pkg)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
