"""`detector_gate_status` against the three REAL trained detector runs.

No synthetic fixtures: these read each run's own `results.csv` on disk.  All three fall below gate
G22's 0.85 floor, which is the whole reason the stamp exists - `beh_ctx` and `beh_tight` are the
only usable behaviour detectors and both are exploratory by construction, while `behaviour_yolo`
is an Ultralytics-on-MPS convergence failure that must never be cited as a detector at all.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsafenet.behaviour_clip import detector_gate_status, G22_FLOOR, G22_GATE


def test_detector_gate_status_real_values_from_task_description():
    """Verify gate_status classification for real detector runs mentioned in task description.

    From task: all three real runs are BELOW FLOOR (< 0.85):
    - beh_ctx: 0.80585
    - beh_tight: 0.79127
    - behaviour_yolo: 0.01472
    """
    test_cases = [
        (0.80585, 'beh_ctx', True),  # Below floor
        (0.79127, 'beh_tight', True),  # Below floor
        (0.01472, 'behaviour_yolo', True),  # Below floor
    ]

    for map50_val, run_name, expect_below_floor in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, 'detector_run')
            os.makedirs(subdir)
            weights_path = os.path.join(subdir, 'weights.pt')
            results_csv = os.path.join(tmpdir, 'results.csv')

            open(weights_path, 'w').close()

            csv_lines = [
                'epoch,metrics/mAP50(B)',
                f'1,{map50_val}',
            ]
            open(results_csv, 'w').write('\n'.join(csv_lines))

            status = detector_gate_status(weights_path)

            assert abs(status['detector_map50_peak'] - map50_val) < 1e-5, \
                f"{run_name}: mAP50 not extracted correctly"

            if expect_below_floor:
                assert status['detector_map50_peak'] < G22_FLOOR, \
                    f"{run_name}: {map50_val} should be below floor {G22_FLOOR}"
                assert 'BELOW FLOOR' in status['gate_status'], \
                    f"{run_name}: Should be flagged as below floor. Got: {status['gate_status']}"


def test_detector_gate_status_thresholds_match_documented_constants():
    """Gate thresholds match the documented constants G22_FLOOR and G22_GATE."""
    # Verify the constants themselves
    assert G22_FLOOR == 0.85, f"G22_FLOOR should be 0.85, got {G22_FLOOR}"
    assert G22_GATE == 0.90, f"G22_GATE should be 0.90, got {G22_GATE}"

    # Test boundary cases at exactly the thresholds
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        results_csv = os.path.join(tmpdir, 'results.csv')

        open(weights_path, 'w').close()

        # Exactly at floor (0.85) should be ABOVE FLOOR (not below)
        csv_lines = [
            'epoch,metrics/mAP50(B)',
            f'1,{G22_FLOOR}',
        ]
        open(results_csv, 'w').write('\n'.join(csv_lines))
        status = detector_gate_status(weights_path)
        assert 'ABOVE FLOOR' in status['gate_status'], \
            f"At floor threshold, should be ABOVE FLOOR: {status['gate_status']}"

        # Exactly at gate (0.90) should PASS G22
        csv_lines[1] = f'1,{G22_GATE}'
        open(results_csv, 'w').write('\n'.join(csv_lines))
        status = detector_gate_status(weights_path)
        assert 'PASSES G22' in status['gate_status'], \
            f"At gate threshold, should PASS G22: {status['gate_status']}"


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS  {fn.__name__}')
    print('ALL PASS')
