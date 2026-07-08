# Benchmark Results — YOLO-Transformer Pipeline on Apple M4

**Generated:** 2026-06-13 13:24  
**Machine:** Phgs-Mac.lan  

## Latency by Stage (p50 / p95 ms, 100 runs, batch=1)

| Config | Backbone | Stats-Agg | Direct4 | Cascade | End-to-End | FPS |
|--------|----------|-----------|---------|---------|------------|-----|
| CPU-FP32 | 8.9/9.7 | 0.2/0.2 | 4.6/4.9 | 17.5/19.5 | 26.8/28.8 | 37.3 |
| CPU-INT8 | 20.4/21.2 | 0.2/0.2 | 2.0/2.3 | 11.3/14.7 | 32.2/35.6 | 31.1 |
| COREML-FP32 | 2.1/2.2 | 0.3/0.3 | 12.3/14.7 | 48.2/51.7 | 50.6/54.1 | 19.8 |
| COREML-INT8 | 35.0/36.3 | 0.3/0.3 | 19.2/20.4 | 53.6/56.1 | 89.0/91.3 | 11.2 |

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