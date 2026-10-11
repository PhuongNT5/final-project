"""Threshold logic of `unsafenet/behaviour_clip.py::detector_gate_status`.

Classifies a detector run's peak `metrics/mAP50(B)` against gate G22: PASSES at >= 0.90, BELOW
GATE/ABOVE FLOOR in [0.85, 0.90), BELOW FLOOR under 0.85.  The CSV fixtures here are synthetic by
necessity - no detector this project has trained reaches 0.85, so the two upper branches have no
real run to exercise them.  They are written to a tmpdir and never into `runs/`.  The real trained
detectors are asserted separately in `test_detector_gate_status_real_runs.py`.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsafenet.behaviour_clip import detector_gate_status, G22_FLOOR, G22_GATE


def test_detector_gate_status_above_gate_threshold():
    """mAP50 >= 0.90 is classified as PASSES G22."""
    # detector_gate_status expects weights at .../subdir/weights.pt
    # and looks for results.csv at ../results.csv (parent dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        results_csv = os.path.join(tmpdir, 'results.csv')

        # Create dummy weights file
        open(weights_path, 'w').close()

        # Create results.csv with high mAP50 in parent dir
        csv_lines = [
            'epoch,train/box_loss,train/cls_loss,val/box_loss,val/cls_loss,metrics/mAP50(B)',
            '1,0.5,0.3,0.4,0.3,0.88',
            '2,0.4,0.2,0.35,0.25,0.92',  # Peak at 0.92 > 0.90
        ]
        open(results_csv, 'w').write('\n'.join(csv_lines))

        status = detector_gate_status(weights_path)

        assert status['detector_map50_peak'] == 0.92, f"mAP50 not extracted correctly: {status}"
        assert 'PASSES G22' in status['gate_status'], \
            f"Should pass gate at 0.92: {status['gate_status']}"


def test_detector_gate_status_between_floor_and_gate():
    """0.85 <= mAP50 < 0.90 is classified as BELOW GATE, ABOVE FLOOR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        results_csv = os.path.join(tmpdir, 'results.csv')

        open(weights_path, 'w').close()

        # mAP50 = 0.87 (between floor 0.85 and gate 0.90)
        csv_lines = [
            'epoch,train/box_loss,metrics/mAP50(B)',
            '1,0.5,0.83',
            '2,0.4,0.87',  # Peak at 0.87
        ]
        open(results_csv, 'w').write('\n'.join(csv_lines))

        status = detector_gate_status(weights_path)

        assert status['detector_map50_peak'] == 0.87, f"mAP50 not extracted: {status}"
        assert 'BELOW GATE, ABOVE FLOOR' in status['gate_status'], \
            f"Should flag as below gate at 0.87: {status['gate_status']}"


def test_detector_gate_status_below_floor():
    """mAP50 < 0.85 is classified as BELOW FLOOR - EXPLORATORY ONLY."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        results_csv = os.path.join(tmpdir, 'results.csv')

        open(weights_path, 'w').close()

        # mAP50 = 0.80 (below floor 0.85)
        csv_lines = [
            'epoch,metrics/mAP50(B)',
            '1,0.75',
            '2,0.80',  # Peak at 0.80 < 0.85
        ]
        open(results_csv, 'w').write('\n'.join(csv_lines))

        status = detector_gate_status(weights_path)

        assert status['detector_map50_peak'] == 0.80, f"mAP50 not extracted: {status}"
        assert 'BELOW FLOOR - EXPLORATORY ONLY' in status['gate_status'], \
            f"Should flag as below floor at 0.80: {status['gate_status']}"


def test_detector_gate_status_missing_results_csv():
    """Missing results.csv returns UNKNOWN status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        open(weights_path, 'w').close()
        # No results.csv created

        status = detector_gate_status(weights_path)

        assert status['detector_map50_peak'] is None
        assert 'UNKNOWN' in status['gate_status'], f"Should indicate unknown: {status['gate_status']}"


def test_detector_gate_status_multiple_epochs_returns_peak():
    """Peak mAP50 across all epochs is used, not just the last epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'detector_run')
        os.makedirs(subdir)
        weights_path = os.path.join(subdir, 'weights.pt')
        results_csv = os.path.join(tmpdir, 'results.csv')

        open(weights_path, 'w').close()

        # Multiple epochs with varying mAP50; peak is in the middle
        csv_lines = [
            'epoch,metrics/mAP50(B)',
            '1,0.80',
            '2,0.88',  # Peak
            '3,0.86',
            '4,0.84',  # Last epoch is not peak
        ]
        open(results_csv, 'w').write('\n'.join(csv_lines))

        status = detector_gate_status(weights_path)

        assert status['detector_map50_peak'] == 0.88, \
            f"Should use peak across all epochs, not last: {status}"


