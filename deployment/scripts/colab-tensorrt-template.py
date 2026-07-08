# Template — upload FP32 ONNXs to Colab T4, run to get TensorRT INT8 latency.
# DO NOT run locally — requires CUDA GPU (NVIDIA T4 or better).
#
# Steps:
#  1. Upload deployment/export/*-fp32.onnx and all8_test_onnx.pt to /content/
#  2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
#  3. Run all cells in order

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
# !pip install onnxruntime-gpu tensorrt --quiet
# import onnxruntime as ort
# print(ort.get_available_providers())  # expect TensorrtExecutionProvider

# ── Cell 2: Paths ─────────────────────────────────────────────────────────────
# import os, json, time, gc
# import numpy as np
# import torch

# ONNX_DIR = "/content/export"
# FEAT_PATH = "/content/all8_test_onnx.pt"
# OUT_JSON  = "/content/benchmarks-tensorrt.json"

# MODELS = {
#     "backbone":  f"{ONNX_DIR}/backbone-fp32.onnx",
#     "direct4":   f"{ONNX_DIR}/direct4-s314-fp32.onnx",
#     "scenario":  f"{ONNX_DIR}/scenario-s7-fp32.onnx",
#     "pair0":     f"{ONNX_DIR}/pair0-s7-fp32.onnx",
#     "pair1":     f"{ONNX_DIR}/pair1-s7-fp32.onnx",
#     "pair2":     f"{ONNX_DIR}/pair2-s7-fp32.onnx",
#     "pair3":     f"{ONNX_DIR}/pair3-s7-fp32.onnx",
# }

# ── Cell 3: TensorRT INT8 calibration + session ───────────────────────────────
# from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType

# class FeatureCalibReader(CalibrationDataReader):
#     """Use first 32 test videos as calibration dataset."""
#     def __init__(self, feat_path, n=32):
#         data = torch.load(feat_path, map_location="cpu")
#         feats = data["features"].numpy().astype("float32")[:n]  # (n,32,16,512)
#         # stats: mean/std/max over spatial dim
#         mean = feats.mean(axis=2); std = feats.std(axis=2); mx = feats.max(axis=2)
#         self.stats = np.concatenate([mean, std, mx], axis=-1)  # (n,32,1536)
#         self.idx = 0
#     def get_next(self):
#         if self.idx >= len(self.stats): return None
#         item = {"features": self.stats[self.idx : self.idx + 1]}
#         self.idx += 1
#         return item

# # Quantize direct4 head to INT8 static with T4 calibration
# quantize_static(
#     MODELS["direct4"],
#     "/content/direct4-trt-int8.onnx",
#     calibration_data_reader=FeatureCalibReader(FEAT_PATH),
#     quant_format="QOperator",
#     weight_type=QuantType.QInt8,
# )

# ── Cell 4: TensorRT EP session ───────────────────────────────────────────────
# TRT_PROVIDERS = [
#     ("TensorrtExecutionProvider", {
#         "trt_fp16_enable": False,
#         "trt_int8_enable": True,
#         "trt_int8_calibration_table_name": "/content/calibration.flatbuffers",
#         "trt_engine_cache_enable": True,
#         "trt_engine_cache_path": "/content/trt_cache",
#     }),
#     "CUDAExecutionProvider",
# ]

# import onnxruntime as ort
# so = ort.SessionOptions()
# bb_sess = ort.InferenceSession(MODELS["backbone"], so, providers=TRT_PROVIDERS)
# print("Backbone EP:", bb_sess.get_providers())

# ── Cell 5: Benchmark loop ────────────────────────────────────────────────────
# WARMUP = 10
# RUNS = 100

# dummy_img = np.random.randn(1, 3, 224, 224).astype("float32")
# inp_name = bb_sess.get_inputs()[0].name

# # Warmup
# for _ in range(WARMUP):
#     bb_sess.run(None, {inp_name: dummy_img})

# # Timed runs
# latencies = []
# for _ in range(RUNS):
#     t0 = time.perf_counter()
#     bb_sess.run(None, {inp_name: dummy_img})
#     latencies.append((time.perf_counter() - t0) * 1000)

# latencies = sorted(latencies)
# result = {
#     "device": "Colab T4",
#     "model": "backbone-fp32-trt",
#     "warmup": WARMUP,
#     "runs": RUNS,
#     "p50_ms": float(np.percentile(latencies, 50)),
#     "p95_ms": float(np.percentile(latencies, 95)),
#     "p99_ms": float(np.percentile(latencies, 99)),
#     "fps": round(1000.0 / float(np.percentile(latencies, 50)), 1),
# }
# print(result)

# ── Cell 6: Save results ──────────────────────────────────────────────────────
# import json
# with open(OUT_JSON, "w") as f:
#     json.dump(result, f, indent=2)
# print(f"Saved to {OUT_JSON}")
# # Download: files.download(OUT_JSON)
