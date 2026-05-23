#!/usr/bin/env bash
# Train + evaluate ClauseGuard across all 5 canonical seeds:
#   42, 123, 456, 789, 1337
#
# Matches the seed list used in paper-a, paper-b, paper-c. Aggregate
# numbers go to ``results/clauseguard_5seeds.json`` via
# ``scripts/aggregate_results.py``.
#
# Usage:
#   bash scripts/run_all_seeds.sh
#
# Optional environment:
#   CLAUSEGUARD_EXTRACTOR=regex|qwen   default: regex
#   CLAUSEGUARD_MAX_TRAIN=10000         dev-friendly cap on FEVER train

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
EXTRACTOR="${CLAUSEGUARD_EXTRACTOR:-regex}"
MAX_TRAIN="${CLAUSEGUARD_MAX_TRAIN:-10000}"

SEEDS=(42 123 456 789 1337)

for seed in "${SEEDS[@]}"; do
    echo "==> Seed $seed"

    # 1. Save verifier (trains on FEVER train, eval on dev).
    python "$ROOT/scripts/save_verifier.py" \
        --seed "$seed" \
        --max-train "$MAX_TRAIN" \
        --extractor "$EXTRACTOR" \
        --output "$ROOT/results/verifiers/verifier_seed_$seed.pkl"

    PKL="$ROOT/results/verifiers/verifier_seed_$seed.pkl"

    # 2. Train+eval on FEVER (per-sample logger).
    python "$ROOT/experiments/train_fever.py" \
        --seed "$seed" \
        --max-train "$MAX_TRAIN" \
        --extractor "$EXTRACTOR"

    # 3. Zero-shot transfer.
    python "$ROOT/experiments/eval_halueval.py" \
        --seed "$seed" --verifier-pickle "$PKL" --extractor "$EXTRACTOR" || true

    python "$ROOT/experiments/eval_factscore.py" \
        --seed "$seed" --verifier-pickle "$PKL" --extractor "$EXTRACTOR" || true

    python "$ROOT/experiments/eval_medhall.py" \
        --seed "$seed" --verifier-pickle "$PKL" --extractor "$EXTRACTOR" || true

    # 4. Recourse.
    python "$ROOT/experiments/recourse_eval.py" \
        --seed "$seed" --verifier-pickle "$PKL" --extractor "$EXTRACTOR" || true

    # 5. Adversarial robustness.
    python "$ROOT/experiments/adversarial_eval.py" \
        --seed "$seed" --verifier-pickle "$PKL" || true

    # 6. SAT receipts.
    python "$ROOT/experiments/sat_receipt_demo.py" \
        --seed "$seed" --verifier-pickle "$PKL" || true
done

python "$ROOT/scripts/aggregate_results.py"

echo "==> All seeds done."
