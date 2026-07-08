# Benchmark Results — YOLO-Transformer Pipeline on Apple M4

**Generated:** 2026-04-21 22:56  
**Machine:** Phgs-Mac.lan  

## Latency by Stage (p50 / p95 ms, 100 runs, batch=1)

| Config | Backbone | Stats-Agg | Direct4 | Cascade | End-to-End | FPS |
|--------|----------|-----------|---------|---------|------------|-----|
| CPU-FP32 | 9.0/10.5 | 0.2/0.2 | 4.7/5.1 | 19.5/28.3 | 28.8/37.3 | 34.7 |
| CPU-INT8 | 20.5/21.7 | 0.3/0.3 | 2.0/2.1 | 11.2/14.0 | 32.0/35.5 | 31.3 |
| COREML-FP32 | 1.8/2.0 | 0.2/0.3 | 12.0/13.5 | 42.7/53.9 | 44.7/56.1 | 22.4 |
| COREML-INT8 | 34.2/50.9 | 0.2/0.3 | 17.4/18.2 | 51.7/64.6 | 86.6/106.5 | 11.5 |

## Accuracy on 4-Class Unsafe Subset

| Config | Direct4 | Cascade | Ensemble |
|--------|---------|---------|----------|
| CPU-FP32 | 55/64 | 51/64 | 56/64 |
| CPU-INT8 | 55/64 | 51/64 | 55/64 |
| COREML-FP32 | 55/64 | 51/64 | 56/64 |
| COREML-INT8 | 55/64 | 51/64 | 55/64 |

## Model Sizes

| File | MB |
|------|----|
| backbone-fp32.onnx | 33.31 |
| backbone-ft-fp32.onnx | 33.3 |
| backbone-ft-int8.onnx | 8.66 |
| backbone-int8.onnx | 8.66 |
| direct4-s314-fp32.onnx | 46.23 |
| direct4-s314-int8.onnx | 11.79 |
| eight-s7-fp32.onnx | 46.23 |
| eight-s7-int8.onnx | 11.79 |
| pair0-s7-fp32.onnx | 7.7 |
| pair0-s7-int8.onnx | 2.03 |
| pair1-s7-fp32.onnx | 7.7 |
| pair1-s7-int8.onnx | 2.03 |
| pair2-s7-fp32.onnx | 7.7 |
| pair2-s7-int8.onnx | 2.03 |
| pair3-s7-fp32.onnx | 7.7 |
| pair3-s7-int8.onnx | 2.03 |
| scenario-s7-fp32.onnx | 46.23 |
| scenario-s7-int8.onnx | 11.79 |

*p50/p95 ms. FPS=1000/p50. e2e=backbone+stats+cascade stages.*