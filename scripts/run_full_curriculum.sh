#!/usr/bin/env bash
# Full SENTINEL-Vision curriculum training pipeline:
#   synthetic injection data -> (optional) contrastive pretrain -> Stage A -> Stage B -> Stage C
#
# Each stage's metrics.csv (all validation metrics, per epoch) and sample
# prediction images (per epoch) are written live by src/training/trainer.py
# to checkpoints/<stage>/metrics.csv and checkpoints/<stage>/images/epoch_NNN/,
# then copied into a timestamped run directory here for archiving.
#
# Epoch counts come from configs/curriculum.yaml (stage_a=10, stage_b=10,
# stage_c=15) with early_stopping_patience=5 — a stage stops early if
# val_recall_harmful hasn't improved in 5 epochs, so actual epoch counts may
# be lower than the configured maximum.
#
# Usage:
#   bash scripts/run_full_curriculum.sh                  # GPU if available, else CPU
#   DEVICE=cpu bash scripts/run_full_curriculum.sh        # force CPU
#   SKIP_CONTRASTIVE=1 bash scripts/run_full_curriculum.sh
#   SKIP_INJECTION=1 bash scripts/run_full_curriculum.sh  # reuse existing synthetic data

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/curriculum_${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/full_training.log"

# Git Bash on Windows (especially when launched from PowerShell) often has a
# different PATH than PowerShell/cmd itself, so `python` can resolve in one
# and not the other even though the same interpreter is installed. Probe a
# few common launcher names instead of assuming `python` is on PATH.
resolve_python() {
    for candidate in python python3 py; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(resolve_python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: no Python interpreter found on PATH in this Git Bash session (tried python, python3, py)." >&2
    echo "Python IS installed on this machine (PowerShell/cmd can already find it via 'where python')" >&2
    echo "-- Git Bash just doesn't inherit that PATH entry. Fix: open PowerShell, run" >&2
    echo "     (Get-Command python).Path" >&2
    echo "   to get its real install directory, then in Git Bash run (adjust the path):" >&2
    echo "     echo 'export PATH=\"/c/Users/<you>/AppData/Local/Programs/Python/Python310:\$PATH\"' >> ~/.bashrc" >&2
    echo "   and open a new Git Bash window before retrying." >&2
    exit 1
fi

DEVICE="${DEVICE:-$($PYTHON_BIN -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null)}"
DEVICE="${DEVICE:-cpu}"
SKIP_INJECTION="${SKIP_INJECTION:-0}"
SKIP_CONTRASTIVE="${SKIP_CONTRASTIVE:-0}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-5}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-data/synthetic_injections}"
N_PER_CATEGORY="${N_PER_CATEGORY:-600}"
N_BENIGN="${N_BENIGN:-2000}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

stage_elapsed() {
    local start="$1" end="$2" label="$3"
    local secs=$(( end - start ))
    printf '%s took %02dh:%02dm:%02ds\n' "$label" $((secs/3600)) $(((secs%3600)/60)) $((secs%60)) | tee -a "$LOG_FILE"
    echo "$secs"
}

copy_stage_outputs() {
    local stage="$1"
    local dest="$RUN_DIR/$stage"
    mkdir -p "$dest"
    [ -f "checkpoints/$stage/metrics.csv" ] && cp "checkpoints/$stage/metrics.csv" "$dest/"
    [ -d "checkpoints/$stage/images" ] && cp -r "checkpoints/$stage/images" "$dest/"
    [ -f "checkpoints/$stage/best.pt" ] && cp "checkpoints/$stage/best.pt" "$dest/"
    log "Copied $stage outputs (metrics.csv, images/, best.pt) to $dest"
}

PIPELINE_START=$(date +%s)
log "=== SENTINEL-Vision full curriculum training ==="
log "Run directory: $RUN_DIR"
log "Device: $DEVICE"

# --- Step 0: synthetic injection dataset ---------------------------------
if [ "$SKIP_INJECTION" = "1" ] && [ -d "$SYNTHETIC_DIR/train" ]; then
    log "Step 0/4: SKIP_INJECTION=1 and $SYNTHETIC_DIR/train exists — reusing existing synthetic data."
else
    log "Step 0/4: Generating synthetic injection dataset (n_per_category=$N_PER_CATEGORY, n_benign=$N_BENIGN)..."
    STEP_START=$(date +%s)
    $PYTHON_BIN -m src.security_wall.cli inject \
        --output-dir "$SYNTHETIC_DIR" \
        --n-per-category "$N_PER_CATEGORY" \
        --n-benign "$N_BENIGN" 2>&1 | tee -a "$LOG_FILE"
    stage_elapsed "$STEP_START" "$(date +%s)" "Synthetic injection generation"
fi

# --- Step 1: contrastive pretraining (optional) --------------------------
if [ "$SKIP_CONTRASTIVE" = "1" ]; then
    log "Step 1/4: SKIP_CONTRASTIVE=1 — skipping contrastive pretraining."
else
    log "Step 1/4: Contrastive temporal pretraining ($CONTRASTIVE_EPOCHS epochs)..."
    STEP_START=$(date +%s)
    $PYTHON_BIN scripts/run_contrastive_pretrain.py \
        --output checkpoints/contrastive_pretrain/temporal_fusion.pt \
        --epochs "$CONTRASTIVE_EPOCHS" \
        --batch-size 16 2>&1 | tee -a "$LOG_FILE"
    stage_elapsed "$STEP_START" "$(date +%s)" "Contrastive pretraining"
    mkdir -p "$RUN_DIR/contrastive_pretrain"
    cp checkpoints/contrastive_pretrain/temporal_fusion.pt "$RUN_DIR/contrastive_pretrain/" 2>/dev/null || true
fi

# --- Step 2: Stage A (destructive only) -----------------------------------
log "Step 2/4: Stage A training (destructive only, up to 10 epochs, device=$DEVICE)..."
STEP_START=$(date +%s)
$PYTHON_BIN -m src.training.train_stageA device="$DEVICE" 2>&1 | tee -a "$LOG_FILE"
STAGE_A_STATUS=${PIPESTATUS[0]:-$?}
stage_elapsed "$STEP_START" "$(date +%s)" "Stage A"
copy_stage_outputs "stage_a"
if [ "$STAGE_A_STATUS" -ne 0 ]; then
    log "Stage A FAILED (exit $STAGE_A_STATUS) — stopping pipeline."
    exit "$STAGE_A_STATUS"
fi

# --- Step 3: Stage B (+ financial, privacy) -------------------------------
log "Step 3/4: Stage B training (destructive+financial+privacy, up to 10 epochs, device=$DEVICE)..."
STEP_START=$(date +%s)
$PYTHON_BIN -m src.training.train_stageB device="$DEVICE" 2>&1 | tee -a "$LOG_FILE"
STAGE_B_STATUS=${PIPESTATUS[0]:-$?}
stage_elapsed "$STEP_START" "$(date +%s)" "Stage B"
copy_stage_outputs "stage_b"
if [ "$STAGE_B_STATUS" -ne 0 ]; then
    log "Stage B FAILED (exit $STAGE_B_STATUS) — stopping pipeline."
    exit "$STAGE_B_STATUS"
fi

# --- Step 4: Stage C (all categories + hard example mining) --------------
log "Step 4/4: Stage C training (all 4 categories, up to 15 epochs, device=$DEVICE)..."
STEP_START=$(date +%s)
$PYTHON_BIN -m src.training.train_stageC device="$DEVICE" 2>&1 | tee -a "$LOG_FILE"
STAGE_C_STATUS=${PIPESTATUS[0]:-$?}
stage_elapsed "$STEP_START" "$(date +%s)" "Stage C"
copy_stage_outputs "stage_c"
if [ "$STAGE_C_STATUS" -ne 0 ]; then
    log "Stage C FAILED (exit $STAGE_C_STATUS) — stopping pipeline."
    exit "$STAGE_C_STATUS"
fi

PIPELINE_END=$(date +%s)
stage_elapsed "$PIPELINE_START" "$PIPELINE_END" "TOTAL pipeline"
log "=== Full curriculum training complete. Final checkpoint: checkpoints/stage_c/best.pt ==="
log "Per-stage metrics.csv and per-epoch sample images archived under: $RUN_DIR"
