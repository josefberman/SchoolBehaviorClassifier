#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: $0 [--skip-calibrate] [--transitions] [--radii R_R R_O R_A]" >&2
  echo "  Example: $0" >&2
  echo "  Example: $0 --transitions" >&2
  echo "  Example: $0 --radii 50 100 150" >&2
  exit 1
}

SKIP_CALIBRATE=0
TRANSITIONS=0
R_R=50
R_O=100
R_A=150
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-calibrate) SKIP_CALIBRATE=1 ;;
    --transitions) TRANSITIONS=1 ;;
    --radii)
      [[ $# -ge 4 ]] || usage
      R_R="$2"
      R_O="$3"
      R_A="$4"
      shift 3
      ;;
    *) usage ;;
  esac
  shift
done

TRANSITION_ARGS=()
if [[ "$TRANSITIONS" -eq 1 ]]; then
  TRANSITION_ARGS=(--transitions)
fi
RADII_ARGS=(--r-r "$R_R" --r-o "$R_O" --r-a "$R_A")

echo "full_run transitions=${TRANSITIONS} radii=${R_R}/${R_O}/${R_A}"

python scripts/tune_features.py --set-radii "${RADII_ARGS[@]}"
if [[ "$SKIP_CALIBRATE" -eq 0 ]]; then
  python scripts/calibrate_baselines.py --source real
else
  echo "Skipping calibrate_baselines.py (reusing results/calibration_report.json)"
fi
python scripts/tune_features.py --target results/calibration_report.json --trials 500 --n-seeds 5 --n-values 20 40
python scripts/tune_features.py --apply-best "${RADII_ARGS[@]}"
python scripts/generate_sims.py --render --video --render-seeds 0 "${TRANSITION_ARGS[@]}"
if [[ "$TRANSITIONS" -eq 1 ]]; then
  python scripts/train_classifier.py --transitions
  python scripts/eval_real.py --transitions
else
  python scripts/train_classifier.py --stable-only
  python scripts/eval_real.py --stable-only
fi
python scripts/plot_confusion.py

echo "All scripts completed for transitions=${TRANSITIONS}"
