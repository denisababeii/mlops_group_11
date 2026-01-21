# tests/__init__.py
"""
Test package initialization.

This file exists mainly to:
1) mark `tests/` as a Python package (useful for imports),
2) provide shared paths that tests can reuse.

Note: In the tests we write below, we mostly use pytest's `tmp_path` to avoid
depending on real data existing on disk.
"""

import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data folder (if used)
