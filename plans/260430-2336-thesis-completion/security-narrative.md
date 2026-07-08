# Security & Privacy Narrative — Edge Deployment

> Source document for Chapter 5 §5.6.X. Quotes from this file should be cited
> back to the canonical numbers in `GROUND_TRUTH.md` and the architecture
> diagram in `figures/arch-diagram-jetson-edge.mmd`.

## 1. Threat model

We consider an industrial surveillance deployment monitoring **multiple camera
streams** for unsafe-behavior detection. The relevant threats are:

| # | Threat | Asset at risk |
|---|---|---|
| T1 | **Raw video exfiltration** — worker imagery leaks through the network or cloud | Operator + worker privacy (GDPR, PIPEDA) |
| T2 | **Model theft** — adversary copies the trained model to replicate or attack it | Intellectual property |
| T3 | **Transport interception** — alerts or metadata sniffed between nodes | Alert integrity, deployment site privacy |
| T4 | **Unauthorised access** — attacker controls the edge node or dashboard | Detection coverage, alert tampering |
| T5 | **False-alarm flooding** — model jitter triggers spurious operator alerts | Operator trust, alarm fatigue |

The architecture we measure (§5.6) addresses T1-T5 directly. Sections 2-6 below
map each control to the implementation.

## 2. Privacy by construction (T1)

The strongest privacy property of the deployment is **architectural**, not policy:

> Raw video frames are **decoded, pre-processed, and inferenced entirely on the
> Jetson edge node**. The only data structure that crosses the network boundary
> is the JSON-Lines prediction stream
> (`{ts, stream_id, frame_n, class, label, probs, head_ms}`).

This means:

1. **Worker faces, gait, location, and behaviour clips never leave the edge.**
   GDPR Article 4 "personal data" — which includes images of identifiable
   persons — stays under the data controller's physical control.
2. **No re-identification surface in transit.** The 38-byte-per-event metadata
   does not contain image data, audio, or biometric features. The class index
   is a 4-bit signal; the probability vector is 4×float ≈ 16 bytes.
3. **The architecture is auditable.** A network monitor on the Jetson uplink
   can verify byte-by-byte that only JSON events leave the device. Compare
   with cloud-inference architectures where frames are POST'd to an inference
   API and re-identification risk is non-trivial to audit.

Measurement evidence (`deployment/artifacts/live-demo-stream.jsonl`):
**3,826 prediction events over a 5-minute 4-stream demo = ~1.99 MB total**,
or ~520 bytes/event including the JSON overhead. By comparison, a single
1080p H.264 frame at 30 FPS is ~50 KB — **two orders of magnitude more data**
and trivially re-identifiable.

## 3. Transport security (T3)

The dashboard pulls the prediction stream via `rsync` over SSH:

```
rsync -az --timeout=5 jetson:~/jetson-stream.jsonl /tmp/jetson-stream.jsonl
```

This composes three controls without any custom code:

| Control | Mechanism | Source |
|---|---|---|
| Authentication | SSH public-key (ED25519) | `~/.ssh/jetson_key` — no password fallback |
| Confidentiality | AES-128-CTR (OpenSSH default cipher) | OpenSSH 9.x defaults |
| Integrity | HMAC-SHA2-256 | OpenSSH 9.x defaults |

The key was bootstrapped by `ssh-copy-id` once, and the alias lives in
`~/.ssh/config`. There is no plaintext channel for prediction events.

For deployments that span untrusted networks (e.g. site → corporate WAN), the
same `rsync` invocation can be tunnelled through a WireGuard or OpenVPN link
without changing the application code — the dashboard already only consumes
the local cache file.

## 4. Model security (T2)

TensorRT engine binaries are **device-specific**: TRT optimises the network
for the exact GPU SM count, memory bandwidth, and clock topology of the
build-time device. The build log includes a fingerprint that the runtime
checks at deserialisation:

```
[06/12/2026-23:37:36] [TRT] [W] Using an engine plan file across different models
of devices is not recommended and is likely to affect performance or even cause errors.
```

In our power-sweep measurements we observed this warning when the engine built
under one `nvpmodel` profile was loaded under another — TRT noticed the device
state had changed and warned. **An engine copied to a different Jetson SKU, or
to a non-Jetson GPU, will refuse to deserialise**.

For an attacker with read access to the edge node, the engine binary alone
is not sufficient to replicate inference on attacker-controlled hardware
without rebuilding from the ONNX source. The ONNX export
(`deployment/export/*-fp32.onnx`) is portable but is the *pre-quantization*
representation; the deployed INT8 calibration data
(`deployment/artifacts/calib-jetson/`) lives separately, so even possession of
the ONNX does not reproduce the deployed engine bit-for-bit.

This is a **defense-in-depth property**, not a hard guarantee — a determined
attacker with full filesystem access can rebuild. But the TRT engine is not
the keys to the kingdom.

## 5. Edge node access (T4)

The Jetson is reachable only by SSH, and only with key-based authentication.
Password authentication was tested early (Day 0) and replaced with
`PubkeyAuthentication yes` + key push via `ssh-copy-id`. Sudo is configured
`NOPASSWD` for the deployment user — this is a **deliberate trade-off**:

- Without `NOPASSWD`, long-running ops like `sudo tegrastats` and
  `sudo nvpmodel` block on a password prompt that defeats `nohup` background
  execution and breaks the soak workflow.
- With `NOPASSWD`, anyone with the SSH key can fully control the device.
  In production, the same control surface would be locked behind a VPN +
  SSH key + jump host topology.

For the thesis demonstration the deployment is single-tenant and physically
controlled. For real industrial use, the SSH key should be in an HSM or
TPM-backed agent and rotated under an inventory management system.

## 6. Decision-policy hardening against false alarms (T5)

The raw per-emission predictions exhibit **77.5% per-frame top-1 accuracy on
3 unsafe streams** (measured in the live demo: 2226/2871 correct emissions),
compared with **87.5% clip-level accuracy** on the same model. The gap is
single-frame jitter — the model output on borderline frames flips between
adjacent classes.

The dashboard mitigates this with two compounding controls (§5.5 of this
chapter and `jetson-live-monitor.py`):

1. **Rolling 3-second aggregation per stream**: nine emissions are averaged
   in probability space, then argmax. This smooths the jitter.
2. **Confidence threshold gating**: aggregated confidence must clear 0.75
   for an ALERT badge; 0.55 for a WARNING badge.

Measurement evidence: at the default thresholds the live demo correctly
flagged the two true-positive streams (S0 walkway violation 76.7%,
S2 overload with forklift 99.4%) as ALERT, and correctly suppressed the
safe-walkway stream (54.9%) below the alert threshold — **0% false alarm
on the safe input**. The misclassified intervention stream (50.4%) sat in
MONITORING; an operator who wants to catch borderline cases can drop the
threshold to 0.55 to surface it as WARNING without flooding with ALERT
badges.

The thresholds are exposed as sidebar sliders, so the operator can tune
sensitivity at deployment time without redeploying the model.

## 7. Summary table — control ↔ threat mapping

| Threat | Control | Implementation | Evidence |
|---|---|---|---|
| T1 raw video exfiltration | Edge inference + metadata-only transport | `jetson-multi-stream-pipeline.py` writes JSONL; dashboard never receives frames | `live-demo-stream.jsonl` content audit |
| T2 model theft | TRT device-locked engines + private INT8 calibration | `deployment/export/jetson/*.engine` rebuilt per Jetson | TRT runtime device-mismatch warning observed |
| T3 transport interception | SSH key-only auth + AES-128-CTR + HMAC | rsync over SSH alias `jetson` | `~/.ssh/config` + `ssh-copy-id` setup |
| T4 unauthorised access | PubkeyAuthentication + NOPASSWD operator account | `/etc/ssh/sshd_config`, `/etc/sudoers.d/phuongnguyen` | Single-tenant, physical control |
| T5 false-alarm flooding | Rolling-window + dual-threshold policy | `jetson-live-monitor.py` aggregate_stream + alert_state | 0 false alarms in live demo at default thresholds |

## 8. What this architecture does NOT claim

- It is **not** a tamper-evident or attestation-bound deployment. A root
  attacker on the Jetson can replace the inference binary.
- It is **not** robust against active model evasion (adversarial patches,
  privacy patches). That is a separate, open research problem.
- It is **not** zero-trust — the dashboard implicitly trusts whatever JSONL
  the Jetson produces. Cryptographic signing of events would close this gap
  with ~50 bytes/event additional metadata and an Ed25519 verify step.

These limitations are surfaced explicitly in the conclusion (§6.3) as future
work. The thesis claim is the narrower and falsifiable one: **the deployed
system preserves raw-video privacy by construction, transmits only ~500 byte
metadata per inference, and supports operator-tunable alerting at the
described latency.**
