#!/usr/bin/env python3
"""Re-grade results/<run>/records.jsonl with the current is_equiv (no regeneration). Recomputes correct,
correct_a/b, agree and the pair pick from the stored predictions; the old file is kept as records.jsonl.bak."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
from twin import is_equiv, pair_pick

for run in sys.argv[1:]:
    path = os.path.join(run, "records.jsonl")
    shutil.copy(path, path + ".bak")
    recs = [json.loads(l) for l in open(path)]
    changed = 0
    for r in recs:
        old = (r.get("correct"), r.get("correct_a"), r.get("correct_b"), r.get("agree"))
        if "pred_a" in r:
            pick, agree = pair_pick(r["pred_a"], r["pred_b"], r["done_a"] != "self", r["done_b"] != "self")
            r.update(pair_pred=pick, pred=pick, agree=agree,
                     correct_a=is_equiv(r["pred_a"], r["gold"]), correct_b=is_equiv(r["pred_b"], r["gold"]))
        r["correct"] = is_equiv(r["pred"], r["gold"])
        changed += old != (r.get("correct"), r.get("correct_a"), r.get("correct_b"), r.get("agree"))
    with open(path, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"{run}: {changed}/{len(recs)} records changed")
