"""Central configuration for the Unsafe-Net R2 pipeline (paper-faithful rewrite).

Reference: Onal & Dandil (2025), Unsafe-Net = YOLO v4 per-frame behaviour detector
+ ConvLSTM temporal classifier.  Paper targets on the 64-clip 4-class unsafe test
subset: 62/64 = 96.88 % top-1, 95.81 % macro (mean per-class recall).
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, 'data')
CACHE_DIR = os.path.join(ROOT, 'cache_r2')          # frame + feature cache (gitignored)
FRAME_DIR = os.path.join(CACHE_DIR, 'frames')
OUT_DIR = os.path.join(ROOT, 'sweep_results', 'r2')

DETECTOR = os.path.join(ROOT, 'sweep_results', 'entity_detector', 'detector_best.pt')
VIEW_MAP = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')
ZONES_DIR = os.path.join(ROOT, 'annotation_guide', 'zones')

# ---- classes -------------------------------------------------------------
# 8 source folders; the 4 unsafe ones (0..3) are the evaluated task.
CLASS_DIRS = [
    '0_safe_walkway_violation',
    '1_unauthorized_intervention',
    '2_opened_panel cover',
    '3_carrying_overload_with_forklift',
    '4_safe_walkway',
    '5_authorized_intervention',
    '6_closed_panel_cover',
    '7_safe_carrying',
]
CLASS_NAMES8 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
                'SafeWalkway', 'AuthorInt', 'ClosedPanel', 'SafeCarry']
UNSAFE = [0, 1, 2, 3]
NAMES4 = CLASS_NAMES8[:4]
# safe counterpart of each unsafe class (used for the 8->4 marginalisation)
SAFE_OF = {0: 4, 1: 5, 2: 6, 3: 7}

# ---- entity detector ontology (Stage A, already trained: mAP@50 = 0.969) --
ENTITY_NAMES = ['Person_Authorized', 'Person_Unauthorized', 'Safe_Walkway', 'Restricted_Zone',
                'Panel_Closed', 'Panel_Opened', 'Forklift_Safe_Load', 'Forklift_Overload']
PERSON_IDS = (0, 1)
PANEL_IDS = (4, 5)
FORKLIFT_IDS = (6, 7)

# ---- sampling ------------------------------------------------------------
T = 32                       # frames cached & used per clip
CACHE_W, CACHE_H = 960, 540  # cached frame size (half of 1920x1080)
DET_IMGSZ = 800
DET_CONF = 0.20
N_BOX = 8                    # max entity tokens per frame

# paper's reported figures, for the comparison tables
PAPER = {
    'full':   {'top1': 62 / 64, 'macro': 0.9581, 'per_class': [1.0, 10 / 11, 12 / 13, 1.0]},
    'nobbox': {'top1': 57 / 64, 'macro': 0.8578, 'per_class': [31 / 32, 9 / 11, 10 / 13, 7 / 8]},
}

for _d in (CACHE_DIR, FRAME_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)
