#!/bin/zsh
# Main cross-insertion experiment, no-think track. Resumable: rerun the script and finished jobs are skipped.
set -e
cd "$(dirname "$0")/.."
export LMS_PARALLEL=4
export LMS_LOAD_CMD="$HOME/.lmstudio/bin/lms load qwen3.6-35b-a3b-mlx --identifier twin -c 16384 --parallel 4 --ttl 360000 -y"
PY=(.venv/bin/python -I)
COMMON=(--think-tokens 2500 --nothink)
TWO_CONDS=${TWO_CONDS:-x_none,x_summary,x_critique,s_wait,s_summary,s_challenge}
TWO_SEEDS=${TWO_SEEDS:-0}

echo "== 1. screen: single x 3 seeds on 80 level-5 problems  $(date +%T)"
$PY src/twin.py --n 80 --conds single --seeds 0,1,2 $COMMON --workers 4 --out results/screen
$PY src/select_problems.py results/screen data/selected_l5.jsonl --easy 10

echo "== 2. single on FRESH seeds 3,4,5 on the selected problems  $(date +%T)"
$PY src/twin.py --data data/selected_l5.jsonl --n 999 --conds single --seeds 3,4,5 $COMMON --workers 4 --out results/main_x

echo "== 3. two-trace conditions $TWO_CONDS, seeds $TWO_SEEDS  $(date +%T)"
$PY src/twin.py --data data/selected_l5.jsonl --n 999 --conds $TWO_CONDS --seeds $TWO_SEEDS $COMMON --workers 2 --out results/main_x
echo "== done  $(date +%T)"
