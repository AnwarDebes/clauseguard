#!/usr/bin/env bash
# Download and cache all four benchmark datasets used by ClauseGuard.
# Network-required; idempotent (re-running re-uses the cache).
#
# Required environment:
#   CLAUSEGUARD_CACHE_DIR  defaults to ~/.cache/clauseguard
#
# Optional environment (lets you skip the FEVER Wikipedia dump
# download, saves ~11 GB but makes ``evidence_sentences`` empty
# for FEVER samples; downstream code falls back to NotEnoughInfo):
#   FEVER_WIKI_DIR=path/to/wiki-pages-dir

set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "==> Downloading FEVER train + labelled_dev..."
$PYTHON - <<'PY'
from clauseguard.data.fever import load_fever
load_fever(split="train")
load_fever(split="dev")
PY

echo "==> Downloading HaluEval QA / Dialogue / Summarization..."
$PYTHON - <<'PY'
from clauseguard.data.halueval import load_halueval
for sub in ("qa", "dialogue", "summarization"):
    load_halueval(sub)
PY

echo "==> Downloading FActScore atomic facts..."
$PYTHON - <<'PY'
from clauseguard.data.factscore import load_factscore
load_factscore()
PY

echo "==> Downloading MedHallBench..."
$PYTHON - <<'PY'
from clauseguard.data.medhall import load_medhall
load_medhall()
PY

echo "==> All datasets cached. Cache directory:"
echo "    ${CLAUSEGUARD_CACHE_DIR:-~/.cache/clauseguard}"
