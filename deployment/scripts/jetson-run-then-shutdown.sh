#!/usr/bin/env bash
# Run ON the Jetson. Runs measurement steps unattended, then powers the board off -- but only if
# every step finished cleanly. Purpose is not speed: it is to leave the board in the lowest-risk
# state when nobody is watching.
#
# Two risks this exists to remove, and nothing else:
#   (a) a shutdown landing mid-write. Each step is waited out to completion; the board is only
#       told to halt afterwards, with sync + a 2-minute abort window.
#   (b) a hung step holding the GPU at load all night. Every step gets a hard STEP_TIMEOUT; a
#       timed-out step is a failure, so the board stays UP (idle draw is 5.3 W, Tj ~55 C) with
#       the GPU released and the log intact for diagnosis on the next boot.
#
# Failure policy is deliberately "stay up": a board left idling costs ~11k VND/month and wears
# nothing, while a board powered off hides the evidence until someone presses the button.
#
# Usage (on the Jetson):
#   bash deployment/scripts/jetson-run-then-shutdown.sh \
#     --step "python3 deployment/scripts/verify-round3-tower-device-parity.py --tower r48-tower-mobileclip_s1" \
#     --step "python3 deployment/scripts/benchmark-r48-clip-pipeline.py --backend tensorrt --mode fp16 --tower r48-tower-mobileclip_s1 --json" \
#     --step "python3 deployment/scripts/benchmark-r48-composed-cadence.py --tower r48-tower-mobileclip_s1 --json"
#
# Detached, so it survives the ssh session dying (which is the normal case overnight):
#   ssh jetson 'cd ~/my-project && setsid nohup bash deployment/scripts/jetson-run-then-shutdown.sh \
#       --step "..." > ~/run-then-shutdown.out 2>&1 &'
#
# Steps wrapped in --guard (default) go through jetson-clean-window-guard.py, so a dirty window
# refuses the measurement instead of producing an unquotable number. --no-guard runs a step raw
# (use for exports/engine builds, which are not measurements).
#
# Abort the halt during its 2-minute window:  sudo shutdown -c
set -uo pipefail

: "${PROJECT_DIR:=$HOME/my-project}"
: "${STEP_TIMEOUT:=5400}"      # 90 min per step; a step past this is treated as hung
: "${SETTLE:=90}"              # guard settle, matches the published protocol
: "${GUARD_TIMEOUT:=900}"      # how long the guard waits for a clean window
: "${HALT_DELAY:=+2}"          # abort window before the board actually halts
: "${LOG:=$HOME/jetson-run-then-shutdown-$(date +%Y%m%dT%H%M%S).log}"
: "${DRY_RUN:=0}"              # 1 = run the steps, print the halt instead of halting

GUARD="deployment/scripts/jetson-clean-window-guard.py"
STEPS=()
GUARDED=()
guard_next=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)     STEPS+=("$2"); GUARDED+=("$guard_next"); guard_next=1; shift 2 ;;
    --guard)    guard_next=1; shift ;;
    --no-guard) guard_next=0; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ ${#STEPS[@]} -gt 0 ]] || { echo "no --step given; nothing to run" >&2; exit 2; }
cd "$PROJECT_DIR" || { echo "no such PROJECT_DIR: $PROJECT_DIR" >&2; exit 2; }

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

log "project $PROJECT_DIR | ${#STEPS[@]} step(s) | step timeout ${STEP_TIMEOUT}s | log $LOG"

failed=0
for i in "${!STEPS[@]}"; do
  step="${STEPS[$i]}"
  if [[ "${GUARDED[$i]}" == "1" ]]; then
    state_out="deployment/artifacts/guard-state-$(date +%Y%m%dT%H%M%S)-step$((i+1)).json"
    mkdir -p deployment/artifacts
    cmd=(python3 "$GUARD" --settle "$SETTLE" --timeout "$GUARD_TIMEOUT" \
         --state-out "$state_out" -- bash -lc "$step")
  else
    cmd=(bash -lc "$step")
  fi

  log "step $((i+1))/${#STEPS[@]} start: $step"
  timeout --signal=TERM --kill-after=120 "$STEP_TIMEOUT" "${cmd[@]}" >>"$LOG" 2>&1
  rc=$?

  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    log "step $((i+1)) HUNG: killed after ${STEP_TIMEOUT}s (rc=$rc). Board stays up."
    failed=1; break
  elif [[ $rc -eq 3 ]]; then
    log "step $((i+1)) UNQUOTABLE: guard says the window turned dirty mid-run. Board stays up."
    failed=1; break
  elif [[ $rc -ne 0 ]]; then
    log "step $((i+1)) FAILED rc=$rc. Board stays up."
    failed=1; break
  fi
  log "step $((i+1)) ok"
done

if [[ $failed -ne 0 ]]; then
  log "NOT halting: a step did not finish cleanly. Inspect $LOG, board left idle."
  exit 1
fi

log "all steps clean; flushing filesystems"
sync
# No measurement process should be left; if one is, something re-spawned and halting is unsafe.
if pgrep -f 'benchmark-r48|benchmark-p3|jetson-measure|trtexec' >/dev/null; then
  log "NOT halting: a measurement process is still running after all steps returned."
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: would run 'sudo shutdown -h $HALT_DELAY'"
  exit 0
fi

log "halting: sudo shutdown -h $HALT_DELAY  (cancel within the window: sudo shutdown -c)"
sudo shutdown -h "$HALT_DELAY" 2>&1 | tee -a "$LOG"
