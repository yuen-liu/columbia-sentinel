#!/bin/bash
# Overnight driver: layer sweep -> full paired embedding -> SAE training.
#
# Stages are idempotent: each checks for its own completion marker and skips
# if already done, so re-running after a crash resumes rather than restarting.
# Embedding resumes mid-stage via progress.json; training resumes mid-stage
# via latest.pt + --resume.
#
# Scope: layer sweep, embedding, and SAE training. Downstream analysis
# stages are not part of this driver.
#
# Usage:  caffeinate -i ./run_overnight.sh

set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/results/logs"
DATA="$ROOT/data"
MANIFEST="$ROOT/results/manifest_combined.csv"
TARGET="$ROOT/config/reg3a_target.fasta"
CHOSEN="$ROOT/results/chosen_layer.json"

mkdir -p "$LOGS" "$DATA"

STATUS="$ROOT/results/STATUS.txt"
note() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"
}

note "=== overnight run started ==="
note "python: $PY"

# ---------------------------------------------------------------- S1: sweep
if [ -f "$CHOSEN" ]; then
    note "S1 layer sweep: SKIP (chosen_layer.json exists)"
else
    note "S1 layer sweep: start"
    "$PY" 01_embed/layer_sweep.py \
        --manifest "$MANIFEST" \
        --target-fasta "$TARGET" \
        --layers 18,19,20,21,22,23,24,25,26,27,28 \
        --n-designs 1500 \
        --output "$ROOT/results/layer_sweep.csv" \
        --chosen-layer-json "$CHOSEN" \
        > "$LOGS/s1_layer_sweep.log" 2>&1
    if [ $? -ne 0 ] || [ ! -f "$CHOSEN" ]; then
        note "S1 FAILED -- falling back to prior-work layer 23 so the night isn't lost"
        note "    (see $LOGS/s1_layer_sweep.log)"
        printf '{"chosen_layer": 23, "reason": "layer_sweep.py failed; fell back to prior-work layer 23", "fallback": true}\n' > "$CHOSEN"
    else
        note "S1 done -> $(cat "$CHOSEN" | tr -d '\n' | cut -c1-200)"
    fi
fi

LAYER=$("$PY" -c "import json;print(json.load(open('$CHOSEN'))['chosen_layer'])")
note "using layer $LAYER"

ACTS_DIR="$DATA/reg3a_layer${LAYER}_paired"
CKPT_DIR="$ROOT/03_train/checkpoints/reg3a_layer${LAYER}_paired"

# ---------------------------------------------------------------- S2: embed
# Completion marker: manifest_combined.csv in the output dir is written only
# after the residue-count assertion passes at the very end of the script.
if [ -f "$ACTS_DIR/manifest_combined.csv" ]; then
    note "S2 embed: SKIP (already complete at $ACTS_DIR)"
else
    note "S2 embed: start (layer $LAYER, ~36.7k designs; resumes if interrupted)"
    for attempt in 1 2 3; do
        "$PY" 01_embed/embed_esmc_paired.py \
            --manifest "$MANIFEST" \
            --output "$ACTS_DIR" \
            --layer "$LAYER" \
            --target-fasta "$TARGET" \
            >> "$LOGS/s2_embed.log" 2>&1
        if [ -f "$ACTS_DIR/manifest_combined.csv" ]; then
            note "S2 done (attempt $attempt)"
            break
        fi
        note "S2 attempt $attempt did not complete; retrying (progress is preserved)"
        sleep 10
    done
    if [ ! -f "$ACTS_DIR/manifest_combined.csv" ]; then
        note "S2 FAILED after 3 attempts -- see $LOGS/s2_embed.log. Stopping."
        exit 1
    fi
fi

# ---------------------------------------------------------------- S3: train
if [ -f "$CKPT_DIR/DONE" ]; then
    note "S3 train: SKIP (DONE marker present)"
else
    note "S3 train: start (d_hidden=16384, k=64, 50 epochs; resumes from latest.pt)"
    RESUME=""
    [ -f "$CKPT_DIR/latest.pt" ] && RESUME="--resume" && note "    resuming from existing latest.pt"
    for attempt in 1 2 3; do
        "$PY" 03_train/train.py \
            --data-dir "$ACTS_DIR" \
            --output-dir "$CKPT_DIR" \
            --d-hidden 16384 \
            --k 64 \
            --auxk 256 \
            --batch-size 4096 \
            --epochs 50 \
            --lr 4e-4 \
            $RESUME \
            >> "$LOGS/s3_train.log" 2>&1
        if grep -q "^Done. Best val_fve" "$LOGS/s3_train.log"; then
            touch "$CKPT_DIR/DONE"
            note "S3 done (attempt $attempt): $(grep '^Done. Best val_fve' "$LOGS/s3_train.log" | tail -1)"
            break
        fi
        note "S3 attempt $attempt stopped early; retrying with --resume"
        RESUME="--resume"
        sleep 10
    done
    if [ ! -f "$CKPT_DIR/DONE" ]; then
        note "S3 did not finish all 50 epochs -- latest.pt holds the furthest epoch reached."
        note "    Re-run this script to continue. See $LOGS/s3_train.log"
    fi
fi

note "=== overnight run finished ==="
note "artifacts:"
note "  layer sweep : results/layer_sweep.csv + results/chosen_layer.json"
note "  activations : $ACTS_DIR"
note "  checkpoints : $CKPT_DIR (best.pt / latest.pt / training_log.csv)"
