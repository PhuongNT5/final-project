"""eval_utils — the thesis's binding evaluation-metric suite (brief S4).

Public entry point: `evaluate_suite` (see suite.py). Individual modules stay
independently importable for callers that only need one metric (existing
`run_full_eval.py`, `run_operating_point.py`, `run_lco_eval.py` import single
functions directly and keep working unchanged).
"""
from eval_utils.suite import evaluate_suite, render_markdown

__all__ = ['evaluate_suite', 'render_markdown']
