"""
run-inference.py — Importable inference helper for ONNX YOLO-Transformer pipeline.

Defines InferenceContext(provider, precision) with:
  - extract_stats(features_NTSC) -> (N,T,1536)
  - head(name, features_NTD) -> logits_NC  (batch=1 loop)
  - direct4_probs, cascade_probs, ensemble_probs
  - stage_times dict recording per-stage latency

Usage (standalone smoke test):
    .venv/bin/python3 deployment/scripts/run-inference.py
"""
import os
import sys
import time
from typing import Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EXPORT_DIR = os.path.join(ROOT, "deployment", "export")

# Cascade pair mapping: pair index → (unsafe_class, safe_class)
PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]

HEAD_FILES = {
    "direct4": "direct4-s314",
    "scenario": "scenario-s7",
    "pair0": "pair0-s7",
    "pair1": "pair1-s7",
    "pair2": "pair2-s7",
    "pair3": "pair3-s7",
    "eight": "eight-s7",
}


class InferenceContext:
    """Loads all ORT sessions once and provides inference methods."""

    def __init__(self, provider: str = "cpu", precision: str = "fp32"):
        """
        provider: 'cpu' or 'coreml'
        precision: 'fp32' or 'int8'
        """
        import onnxruntime as ort

        self.provider = provider
        self.precision = precision
        self.stage_times: Dict[str, float] = {}

        ep_map = {
            "cpu": ["CPUExecutionProvider"],
            "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        }
        providers = ep_map.get(provider, ["CPUExecutionProvider"])

        avail = ort.get_available_providers()
        if "CoreMLExecutionProvider" not in avail and provider == "coreml":
            print("[WARN] CoreML EP not available — falling back to CPU")
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.log_severity_level = 3  # suppress ORT warnings

        backbone_path = os.path.join(EXPORT_DIR, f"backbone-{precision}.onnx")
        assert os.path.exists(backbone_path), f"Missing: {backbone_path}"
        self.backbone_sess = ort.InferenceSession(backbone_path, so, providers=providers)
        self._actual_eps = self.backbone_sess.get_providers()

        self.head_sessions: Dict[str, object] = {}
        for key, fname in HEAD_FILES.items():
            path = os.path.join(EXPORT_DIR, f"{fname}-{precision}.onnx")
            if os.path.exists(path):
                self.head_sessions[key] = ort.InferenceSession(path, so, providers=providers)

        print(f"InferenceContext[{provider}/{precision}] actual EPs: {self._actual_eps}")

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame → (1,3,224,224) float32. PIL BILINEAR to match training."""
        import cv2
        from PIL import Image
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        normalized = (arr - MEAN) / STD
        return normalized.transpose(2, 0, 1)[np.newaxis]

    def backbone_forward(self, image_1CHW: np.ndarray) -> np.ndarray:
        """image (1,3,224,224) → tokens (1,16,512)."""
        inp_name = self.backbone_sess.get_inputs()[0].name
        return self.backbone_sess.run(None, {inp_name: image_1CHW})[0]

    def extract_stats(self, features_NTSC: np.ndarray) -> np.ndarray:
        """(N,T,S,C) → (N,T,D) where D=C*3=1536 via mean/std/max over dim 2."""
        t0 = time.perf_counter()
        mean = features_NTSC.mean(axis=2)
        std = features_NTSC.std(axis=2)
        mx = features_NTSC.max(axis=2)
        result = np.concatenate([mean, std, mx], axis=-1)
        self.stage_times["stats_agg"] = (time.perf_counter() - t0) * 1000
        return result

    def head(self, name: str, features_NTD: np.ndarray) -> np.ndarray:
        """(N,T,D) → (N,C) logits via batch=1 loop."""
        sess = self.head_sessions[name]
        inp_name = sess.get_inputs()[0].name
        results = []
        for i in range(len(features_NTD)):
            inp = features_NTD[i : i + 1]  # (1,T,D)
            out = sess.run(None, {inp_name: inp})[0]  # (1,C)
            results.append(out[0])
        return np.stack(results)  # (N,C)

    def direct4_probs(self, features_NTSC: np.ndarray) -> np.ndarray:
        """→ (N,4) softmax probs for 4-class unsafe classification."""
        stats = self.extract_stats(features_NTSC)  # (N,T,1536)
        t0 = time.perf_counter()
        logits = self.head("direct4", stats)
        self.stage_times["direct4_head"] = (time.perf_counter() - t0) * 1000
        return _softmax(logits)

    def cascade_probs(self, features_NTSC: np.ndarray) -> np.ndarray:
        """→ (N,4) via scenario × pairwise product (lines 472-480 of run_pairwise_contrastive)."""
        stats = self.extract_stats(features_NTSC)

        t0 = time.perf_counter()
        scenario_logits = self.head("scenario", stats)
        scenario_p = _softmax(scenario_logits)  # (N,4) — 4 scenarios

        pair_cols = []
        for i in range(4):
            key = f"pair{i}"
            pair_logits = self.head(key, stats)
            pair_p = _softmax(pair_logits)  # (N,2): col 1 = P(unsafe) — label 1 was unsafe in training
            pair_cols.append(pair_p[:, 1:2])

        pair_p_combined = np.concatenate(pair_cols, axis=1)  # (N,4)
        cascade = scenario_p * pair_p_combined
        norm = cascade.sum(axis=1, keepdims=True) + 1e-8
        self.stage_times["cascade_heads"] = (time.perf_counter() - t0) * 1000
        return cascade / norm  # (N,4)

    def ensemble_probs(self, features_NTSC: np.ndarray) -> np.ndarray:
        """Average direct4 + cascade, then L1-normalize."""
        d4 = self.direct4_probs(features_NTSC)
        cas = self.cascade_probs(features_NTSC)
        combined = (d4 + cas) / 2.0
        norm = combined.sum(axis=1, keepdims=True) + 1e-8
        return combined / norm


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    # Smoke test
    ctx = InferenceContext("cpu", "fp32")
    dummy = np.random.randn(2, 32, 16, 512).astype(np.float32)
    p = ctx.ensemble_probs(dummy)
    print(f"Ensemble probs shape: {p.shape}")
    print(f"Stage times: {ctx.stage_times}")
    print("run-inference.py smoke test passed.")
