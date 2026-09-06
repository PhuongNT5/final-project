"""Selective classification: risk-coverage curve + AURC (El-Yaniv & Wiener 2010;
Geifman & El-Yaniv 2017, "Selective Classification for Deep Neural Networks").

Confidence-ranked deferral: rank clips by model confidence, keep the top-c
fraction as auto-decisions, defer the rest (e.g. to a human reviewer). This is
a DIAGNOSTIC, not a finite-sample coverage guarantee (contrast with
conformal.py) -- it answers "how much does error drop if we abstain on the
least-confident X%," not "what is the guaranteed error rate on new data."
"""
from __future__ import annotations
import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz  # numpy>=2.0 renamed trapz


def _confidence(probs):
    return np.asarray(probs, dtype=float).max(axis=1)


def risk_coverage_curve(probs, labels, preds=None):
    """Rank clips by confidence descending; at each achievable coverage
    c=m/n (m=1..n), selective risk = error rate on the m most-confident
    predictions. Returns (coverage, risk), both length n, coverage[i]=(i+1)/n."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1) if preds is None else np.asarray(preds)
    conf = _confidence(probs)
    order = np.argsort(-conf, kind='stable')
    correct = (preds == labels).astype(float)[order]
    n = len(labels)
    m = np.arange(1, n + 1)
    coverage = m / n
    risk = 1.0 - np.cumsum(correct) / m
    return coverage, risk


def oracle_risk_coverage_curve(probs, labels, preds=None):
    """Best-possible ranking: correct predictions ranked first. Isolates the
    irreducible error floor from confidence-ranking failure; AURC - AURC* =
    excess AURC = the cost of the model's confidence being an imperfect ranker
    of its own correctness."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1) if preds is None else np.asarray(preds)
    correct = (preds == labels).astype(float)
    order = np.argsort(-correct, kind='stable')
    n = len(labels)
    m = np.arange(1, n + 1)
    return m / n, 1.0 - np.cumsum(correct[order]) / m


def aurc(coverage, risk):
    """Area under the risk-coverage curve, trapezoid rule, extending the curve
    down to (0, risk[0]) as the c->0 limit (Geifman & El-Yaniv 2017 convention)."""
    coverage = np.asarray(coverage, dtype=float)
    risk = np.asarray(risk, dtype=float)
    c0 = np.concatenate([[0.0], coverage])
    r0 = np.concatenate([[risk[0]], risk])
    return float(_trapz(r0, c0))


def excess_aurc(probs, labels, preds=None):
    """{'aurc', 'aurc_oracle', 'excess_aurc'}. excess_aurc >= 0 in expectation;
    0 would mean confidence already ranks clips as well as an oracle could."""
    cov, risk = risk_coverage_curve(probs, labels, preds)
    ocov, orisk = oracle_risk_coverage_curve(probs, labels, preds)
    a = aurc(cov, risk)
    a_star = aurc(ocov, orisk)
    return {'aurc': a, 'aurc_oracle': a_star, 'excess_aurc': float(a - a_star)}


def risk_at_coverage(probs, labels, coverage_target, preds=None):
    """Selective risk when auto-deciding the coverage_target most-confident
    fraction and deferring the rest (e.g. coverage_target=0.8 -> defer the
    least-confident 20% of clips to a human reviewer)."""
    cov, risk = risk_coverage_curve(probs, labels, preds)
    n = len(cov)
    m = max(1, min(n, int(round(coverage_target * n))))
    return {'coverage': float(cov[m - 1]), 'risk': float(risk[m - 1]), 'm': m, 'n': n}


def coverage_at_risk(probs, labels, risk_target, preds=None):
    """Largest coverage on this confidence-ranked curve whose selective risk is
    <= risk_target (deferral threshold needed to guarantee a target error rate
    on the accepted clips). risk is not assumed monotone in coverage, so this
    is the literal largest-index match, not the first violation point."""
    cov, risk = risk_coverage_curve(probs, labels, preds)
    ok = np.where(risk <= risk_target)[0]
    if len(ok) == 0:
        return {'coverage': None, 'risk': None,
                'note': f'risk<={risk_target} unreachable at any coverage'}
    i = int(ok.max())
    return {'coverage': float(cov[i]), 'risk': float(risk[i])}
