"""Canonical result schema + markdown-table renderer (brief S4.4h).

Every reported number must carry n, a CI (or None if not applicable), seed count,
and protocol tag — so results paste straight into the thesis without ambiguity
about what was measured. This module is the single place that shape is defined.
"""
from __future__ import annotations
import json


def make_record(value, ci=None, n=None, seed_count=1, protocol_tag='unspecified', **extra):
    """One canonical result entry. `ci` is (lo, hi) or None. `extra` holds any
    metric-specific fields (e.g. per-class breakdowns) without breaking the schema."""
    return {
        'value': value,
        'ci95': [float(ci[0]), float(ci[1])] if ci is not None else None,
        'n': int(n) if n is not None else None,
        'seed_count': int(seed_count),
        'protocol_tag': str(protocol_tag),
        **extra,
    }


def validate_record(record):
    """Raise if a record is missing any of the mandatory S4.4h fields."""
    required = ('value', 'ci95', 'n', 'seed_count', 'protocol_tag')
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f'result record missing required fields: {missing}')
    return True


def to_markdown_table(records: dict, columns=('value', 'ci95', 'n', 'seed_count', 'protocol_tag')):
    """records: {metric_name: record_dict (from make_record)}. Renders a markdown
    table with one row per metric — pastes directly into the thesis results chapter."""
    hdr = '| metric | ' + ' | '.join(columns) + ' |'
    sep = '|' + '---|' * (len(columns) + 1)
    rows = [hdr, sep]
    for name, rec in records.items():
        cells = []
        for c in columns:
            v = rec.get(c)
            if isinstance(v, float):
                v = f'{v:.4f}'
            elif isinstance(v, list):
                v = '[' + ', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in v) + ']'
            cells.append(str(v))
        rows.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def save_json(result: dict, path: str):
    """Serialise a full evaluate_suite() result to disk (pretty-printed)."""
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return path
