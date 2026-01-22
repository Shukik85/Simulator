"""Pytest configuration.

Goal: make `import hydrosim` work reliably when running tests without installing
package (editable install).

This repo uses a flat layout (hydrosim/ at repo root). Some environments run
pytest with a working directory where repo root isn't on sys.path, leading to
`ModuleNotFoundError: hydrosim`.

This conftest ensures repo root is on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
