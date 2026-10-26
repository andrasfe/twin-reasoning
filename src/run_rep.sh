#!/bin/zsh
# Replication of the sparse-exchange lead (DESIGN.md 6.3), same problems and single control in results/main_x.
# Resumable.
#   A. s_challenge@4 and its peer-framed twin x_critique@4, seeds 1 and 2 (seed 0 is already done)
#   B. cadence @8 (~1 exchange per trace), s_challenge, seed 0
set -e
cd "$(dirname "$0")/.."
export LMS_PARALLEL=4
export LMS_LOAD_CMD="$HOME/.lmstudio/bin/lms load qwen3.6-35b-a3b-mlx --identifier twin -c 16384 --parallel 4 --ttl 360000 -y"
PY=(.venv/bin/python -I)
COMMON=(--data data/selected_l5.jsonl --n 999 --think-tokens 2500 --nothink --workers 2 --out results/main_x)

echo "== A. s_challenge@4, x_critique@4, seeds 1,2  $(date +%T)"
$PY src/twin.py --conds s_challenge@4,x_critique@4 --seeds 1,2 $COMMON
echo "== B. s_challenge@8, seed 0  $(date +%T)"
$PY src/twin.py --conds s_challenge@8 --seeds 0 $COMMON
echo "== done  $(date +%T)"
