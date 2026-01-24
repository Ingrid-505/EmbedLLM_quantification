#!/usr/bin/env bash
set -euo pipefail

# ===========================
# run_parallel_evals.sh
# Read model.csv (column: model) and launch eval_one.py jobs.
# Default: 2 parallel workers, each worker uses a GPU pair (0,1) and (2,3).
# ===========================

CSV_PATH="${1:-wuyang_models_zhuoying.csv}"          # first arg: path to csv
MODEL_COL="${MODEL_COL:-model_name}"     # column name in csv; override via env MODEL_COL
EVAL_PY="${EVAL_PY:-run_one.py}"   # path to eval_one.py
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TASKS="${TASKS:-arc_challenge,hellaswag,mmlu,truthfulqa_mc1}"

# GPU groups for each worker (each group is what you pass to CUDA_VISIBLE_DEVICES)
# Edit if you want different pairing.
GPU_GROUPS=(
  "4,5"
  "6,7"
)

# If DRY_RUN=1, only print commands without executing
DRY_RUN="${DRY_RUN:-0}"

# -------- helpers --------
log() { echo "[$(date +'%F %T')] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

need_cmd python3

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  exit 1
fi

if [[ ! -f "$EVAL_PY" ]]; then
  echo "eval_one.py not found: $EVAL_PY (set EVAL_PY=... if needed)" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract models with python's csv module (handles header properly).
# Assumption: model names do not contain embedded newlines.
mapfile -t MODELS < <(CSV_PATH="$CSV_PATH" MODEL_COL="$MODEL_COL" python3 - <<'PY'
import csv, os, sys

csv_path = os.environ.get("CSV_PATH")
col = os.environ.get("MODEL_COL", "model")

if not csv_path:
    raise SystemExit("CSV_PATH env is empty (bug in launcher).")

with open(csv_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    if r.fieldnames is None:
        raise SystemExit("CSV has no header.")
    if col not in r.fieldnames:
        raise SystemExit(f"Column '{col}' not found. Available: {r.fieldnames}")
    out = []
    for row in r:
        m = (row.get(col) or "").strip()
        if m:
            out.append(m)
    for m in out:
        print(m)
PY
)


if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models found in $CSV_PATH column '$MODEL_COL'." >&2
  exit 1
fi

log "Found ${#MODELS[@]} models in $CSV_PATH (column: $MODEL_COL)"
log "GPU groups: ${GPU_GROUPS[*]}"
log "batch_size=$BATCH_SIZE tasks=$TASKS output_dir=$OUTPUT_DIR"
[[ "$DRY_RUN" == "1" ]] && log "DRY_RUN=1 (printing commands only)"

# Worker function: process models with indices i where i % N == worker_id
worker() {
  local worker_id="$1"
  local gpu_group="$2"
  local n_workers="$3"

  local i=0
  for model_name in "${MODELS[@]}"; do
    if (( i % n_workers == worker_id )); then

      # --- skip if result exists (same naming: / -> --) ---
      local safe_name="${model_name//\//--}"
      local result_path="$OUTPUT_DIR/${safe_name}.json"
      if [[ -f "$result_path" ]]; then
        echo "[SKIP] Exists: $result_path"
        ((i+=1))
        continue
      fi
      # ---------------------------------------------------

      local cmd=(python3 "$EVAL_PY"
        --model_name "$model_name"
        --output_dir "$OUTPUT_DIR"
        --batch_size "$BATCH_SIZE"
        --tasks "$TASKS"
      )

      if [[ "$DRY_RUN" == "1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=$gpu_group ${cmd[*]}"
      else
        log "Worker $worker_id on GPUs [$gpu_group] -> $model_name"
        CUDA_VISIBLE_DEVICES="$gpu_group" "${cmd[@]}" || true
      fi
    fi
    ((i+=1))
  done
}


N_WORKERS="${#GPU_GROUPS[@]}"

# Export for python extractor
export CSV_PATH MODEL_COL

# Launch workers in parallel
pids=()
for wid in "${!GPU_GROUPS[@]}"; do
  gpu="${GPU_GROUPS[$wid]}"
  worker "$wid" "$gpu" "$N_WORKERS" &
  pids+=("$!")
done

# Wait for all workers
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [[ "$fail" -ne 0 ]]; then
  log "Some workers exited with non-zero status."
  exit 1
fi

log "All done."
