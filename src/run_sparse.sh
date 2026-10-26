#!/bin/zsh
# Follow-up to run_main.sh, same problems and single control in results/main_x. Resumable.
#   A. sparse exchange: inserts after every 4th round (~700 tokens) instead of every round (~200 tokens)
#   B. second seed for the dense conditions (s_wait left out: -0.5 vs single on seed 0, no replication needed)
set -e
cd "$(dirname "$0")/.."
export LMS_PARALLEL=4
export LMS_LOAD_CMD="$HOME/.lmstudio/bin/lms load qwen3.6-35b-a3b-mlx --identifier twin -c 16384 --parallel 4 --ttl 360000 -y"
PY=(.venv/bin/python -I)
COMMON=(--data data/selected_l5.jsonl --n 999 --think-tokens 2500 --nothink --workers 2 --out results/main_x)
SPARSE=${SPARSE:-s_wait@4,s_challenge@4,s_summary@4,x_critique@4,x_summary@4}
REPL=${REPL:-x_none,x_summary,x_critique,s_summary,s_challenge}

echo "== A. sparse exchange $SPARSE, seed 0  $(date +%T)"
$PY src/twin.py --conds $SPARSE --seeds 0 $COMMON
echo "== B. dense conditions $REPL, seed 1  $(date +%T)"
$PY src/twin.py --conds $REPL --seeds 1 $COMMON
echo "== done  $(date +%T)"
