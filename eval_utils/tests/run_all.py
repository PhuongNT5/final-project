"""Minimal test runner (no pytest in the project venv). Discovers every test_*
function in every eval_utils/tests/test_*.py module and runs it, printing PASS/FAIL.

Usage: .venv/bin/python -m eval_utils.tests.run_all
"""
from __future__ import annotations
import importlib
import pkgutil
import sys
import traceback

import eval_utils.tests as _pkg


def run_module(module):
    """Run every test_* function defined in `module`. Returns 0 if all pass."""
    failures = 0
    ran = 0
    for name in dir(module):
        if not name.startswith('test_'):
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue
        ran += 1
        try:
            fn()
            print(f"  PASS  {module.__name__}.{name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {module.__name__}.{name}")
            traceback.print_exc()
    if ran == 0:
        print(f"  (no test_* functions found in {module.__name__})")
    return 1 if failures else 0


def main():
    total_fail = 0
    for _, mod_name, _ in pkgutil.iter_modules(_pkg.__path__, prefix='eval_utils.tests.'):
        if not mod_name.endswith('.run_all') and 'test_' in mod_name:
            mod = importlib.import_module(mod_name)
            print(f"--- {mod_name} ---")
            total_fail += run_module(mod)
    print()
    print("ALL PASS" if total_fail == 0 else f"FAILURES in {total_fail} module(s)")
    return 1 if total_fail else 0


if __name__ == '__main__':
    sys.exit(main())
