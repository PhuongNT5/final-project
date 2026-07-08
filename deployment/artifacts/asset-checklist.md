# Thesis Chapter 5 — Asset Checklist

All assets generated from measured JSON data or frozen ONNX models. None hand-crafted.

| # | Asset | Source | Thesis Target | Status |
|---|-------|--------|---------------|--------|
| 1 | Architecture diagram | `figures/arch-diagram.mmd` | §5.2 Figure 5.1 | done |
| 2 | Latency by stage (stacked bars) | `figures/latency-by-stage.png` | §5.4 Figure 5.2 | done |
| 3 | Accuracy by config (bar chart) | `figures/accuracy-by-config.png` | §5.4 Figure 5.3 | done |
| 4 | Model size comparison (FP32/INT8) | `figures/model-size-comparison.png` | §5.4 Figure 5.4 | done |
| 5 | FPS vs stream count | `figures/fps-vs-streams.png` | §5.5 Figure 5.5 | done |
| 6 | ONNX graph screenshot (direct4) | `figures/onnx-graph-direct4.png` | §5.3 Figure 5.6 | manual: `netron deployment/export/direct4-s314-fp32.onnx` |
| 7 | Benchmarks table | `benchmarks.md` | §5.4 Table 5.1 | done |
| 8 | Accuracy report | `accuracy-report.txt` | §5.4 Table 5.2 | done |
| 9 | Multi-stream results (JSON → table) | `multi-stream-results.json` | §5.5 Table 5.3 | done (manual formatting in thesis) |
| 10 | Parity report (FP32/INT8 drift) | `parity-report.txt` | §5.3 Table 5.4 | done |
| 11 | Jetson projection table | `plans/reports/researcher-track-b-jetson-orin-projection.md` | §5.6 Table 5.5 | done (projected, unverified) |

## Commands to regenerate

```bash
.venv/bin/python3 deployment/scripts/benchmark-harness.py
.venv/bin/python3 deployment/scripts/multi-stream-pipeline.py --streams 1 2 3 4 --source <4 videos>
.venv/bin/python3 deployment/scripts/generate-figures.py
```

## Notes

- PNG at 150 dpi is sufficient for LaTeX thesis compile; upscale to 300 dpi only if printer requires.
- Mermaid diagram renders in GitHub/GitLab preview. To convert to PNG: `mmdc -i figures/arch-diagram.mmd -o figures/arch-diagram.png` (requires `@mermaid-js/mermaid-cli` npm package).
- Netron screenshot is manual; no headless API. Alternative: `onnx.helper.printable_graph(model.graph)` renders text summary.
