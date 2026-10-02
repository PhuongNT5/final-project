"""evaluate_suite(): schema shape + JSON round-trip on synthetic data."""
import json
import numpy as np

from eval_utils.suite import evaluate_suite, render_markdown
from eval_utils.report_schema import validate_record, save_json

CLASSES = ['A', 'B', 'C', 'D']


def _toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=40)
    probs = rng.dirichlet([1, 1, 1, 1], size=40)
    return y, probs


def test_every_headline_metric_is_a_valid_schema_record():
    y, probs = _toy()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='unit-test-40', n_seeds=3, n_boot=200)
    for key in ('top1', 'macro_accuracy', 'macro_f1', 'balanced_accuracy',
               'cohen_kappa', 'matthews_mcc'):
        validate_record(res[key])
        assert res[key]['protocol_tag'] == 'unit-test-40'
        assert res[key]['n'] == 40
        assert res[key]['seed_count'] == 3


def test_per_class_and_safety_and_operating_point_present():
    y, probs = _toy()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='unit-test-40', n_boot=200)
    assert set(res['per_class'].keys()) == set(CLASSES)
    for c in CLASSES:
        assert 'fnr_miss_rate' in res['per_class'][c]
    assert 'total_cost' in res['safety_cost'] and 'cost_matrix' in res['safety_cost']
    assert 'tau' in res['operating_point_safety_recall']
    assert res['calibration']['binning_scheme']


def test_comparison_block_with_per_clip_predictions():
    y, probs = _toy()
    other_pred = probs.argmax(1).copy()
    other_pred[:5] = (other_pred[:5] + 1) % 4  # force some discordance
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='unit-test-40', n_boot=200,
                         compare_to={'name': 'other-model', 'correct': int((other_pred == y).sum()),
                                     'n': 40, 'pred': other_pred})
    assert 'mcnemar_exact' in res['comparison']
    assert 'mcnemar_discordance_bound' in res['comparison']
    assert 'diagnostic_two_proportion_z' in res['comparison']
    # the unpaired test must never be exposed under an unqualified key
    assert 'two_proportion_z' not in res['comparison']
    assert 'diagnostic_warning' in res['comparison']


def test_result_is_json_serialisable_and_markdown_renders():
    y, probs = _toy()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='unit-test-40', n_boot=200)
    text = json.dumps(res, default=float)
    assert len(text) > 0
    md = render_markdown(res)
    assert md.startswith('| metric |')
    assert 'top1' in md


def test_save_json_round_trip(tmp_path=None):
    import tempfile, os
    y, probs = _toy()
    res = evaluate_suite(y, probs, CLASSES, protocol_tag='unit-test-40', n_boot=200)
    with tempfile.TemporaryDirectory() as d:
        p = save_json(res, os.path.join(d, 'result.json'))
        loaded = json.load(open(p))
        assert loaded['protocol_tag'] == 'unit-test-40'


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
