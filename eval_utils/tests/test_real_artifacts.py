"""Validate evaluate_suite() against REAL artifacts already committed in this repo
(no synthetic/fake data). Numbers cross-checked against independently-computed
references (verify_against_csv.py's asserted 56/64 + 89.28%, and hand-run sklearn
equivalents of run_full_eval.py's inline logic on the current .pt file)."""
import csv
import os
import numpy as np
import torch

from eval_utils.suite import evaluate_suite

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def _load_csv():
    path = os.path.join(ROOT, 'deployment', 'artifacts', 'per-clip-logits-test64.csv')
    with open(path) as f:
        rows = [ln for ln in f if not ln.lstrip().startswith('#')]
    rd = csv.DictReader(rows)
    yt, probs = [], []
    for r in rd:
        yt.append(int(r['true_label']))
        probs.append([float(r[f'ensemble_prob_{c}']) for c in range(4)])
    return np.array(yt), np.array(probs)


def test_reproduces_committed_64clip_headline():
    """Same artifact + same assertions as eval_utils/verify_against_csv.py: 56/64
    top-1, 89.28% macro (mean per-class recall)."""
    y, probs = _load_csv()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='same-dist-64', n_seeds=5)
    assert res['top1']['correct'] == 56, res['top1']['correct']
    assert abs(res['top1']['value'] * 100 - 87.5) < 1e-6
    assert abs(res['macro_accuracy']['value'] * 100 - 89.28) < 0.05


def test_64clip_perclass_and_confusion_shape():
    y, probs = _load_csv()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='same-dist-64')
    # OpenedPanel/CarryOverload are 100% recall on this artifact (verify_against_csv.py)
    assert abs(res['per_class']['OpenedPanel']['recall'] - 1.0) < 1e-9
    assert abs(res['per_class']['CarryOverload']['recall'] - 1.0) < 1e-9
    cm = np.array(res['confusion_matrix'])
    assert cm.shape == (4, 4)
    assert cm.sum() == 64


def test_64clip_comparison_vs_paper_full_pipeline():
    """56/64 (ours) vs paper's declared 62/64 -> discordance bound + two-prop z,
    reproducing the sign/direction established in the S3b audit."""
    y, probs = _load_csv()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='same-dist-64',
                         compare_to={'name': 'paper-full-62-64', 'correct': 62, 'n': 64})
    bound = res['comparison']['mcnemar_discordance_bound']
    assert bound['diff'] == 56 - 62 == -6
    assert bound['favours_b'] is True  # favours the paper (b_is='paper-full-62-64')
    assert abs(bound['min_p_over_splits'] - 0.03125) < 1e-9  # matches audit table k=56


def test_reproduces_relation_tokens_s1a_inline_sklearn_numbers():
    """Cross-check against an independent, hand-run sklearn computation of the exact
    formulas run_full_eval.py uses inline, on the CURRENT logits_s1a.pt (the
    committed full_eval.json is stale relative to this file, confirmed via mtimes
    and a raw argmax recount — see f1-metric-suite.md)."""
    path = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'logits_s1a.pt')
    d = torch.load(path, weights_only=False)
    P = d['ens_softmax'].numpy(); y = d['labels'].numpy()
    res = evaluate_suite(y, P, CLASSES, protocol_tag='same-dist-64-s1a')

    from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
    pred = P.argmax(1)
    assert abs(res['top1']['value'] - (pred == y).mean()) < 1e-9
    assert abs(res['macro_f1']['value'] - f1_score(y, pred, average='macro')) < 1e-6
    assert abs(res['balanced_accuracy']['value'] - balanced_accuracy_score(y, pred)) < 1e-6
    assert abs(res['cohen_kappa']['value'] - cohen_kappa_score(y, pred)) < 1e-6
    assert abs(res['matthews_mcc']['value'] - matthews_corrcoef(y, pred)) < 1e-6


def test_reproduces_brief_kfold_516_headline():
    """k-fold 516-clip pooled OOF softmax, canonical --repeats 10 run (50 folds).

    Pinned to sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log:
      pooled 421/516 = 81.59% top-1, macro 84.24%.

    NOTE two DISTINCT top-1 quantities that must not be swapped:
      * pooled OOF top-1  = 421/516 = 81.6%  <- what this artifact contains and what is asserted here
      * fold-mean top-1   = 80.8 +/- 4.0%    <- the number quoted with the NB-corrected CI in Ch7
    The superseded single-repeat 5-fold run gave pooled 417/516 = 80.81%, which happened to coincide
    with the canonical *fold-mean* 80.8% - that coincidence is why the two were previously conflated.
    """
    path = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'kfold_oof_softmax_yolo26app_.pt')
    d = torch.load(path, weights_only=False)
    P = d['ens_softmax'].numpy(); y = d['labels'].numpy()
    assert len(y) == 516
    res = evaluate_suite(y, P, CLASSES, protocol_tag='kfold-516', n_seeds=5, n_boot=1000)
    assert abs(res['top1']['value'] * 100 - 81.59) < 0.05
    assert abs(res['macro_accuracy']['value'] * 100 - 84.24) < 0.05
    assert res['top1']['correct'] == 421


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
