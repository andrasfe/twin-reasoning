#!/usr/bin/env python3
"""Pick the evaluation set from a screening run of `single` (python src/select_problems.py results/screen out.jsonl).

Keeps every problem the single model does not solve on all screening seeds ("contested") plus a random sample of
the always-solved ones ("easy"), so harm on easy problems (herding onto a wrong answer) stays visible. The main run
must evaluate `single` on FRESH seeds: the screening seeds were used for selection and would regress to the mean.
"""
import argparse, collections, json, os, random
ap = argparse.ArgumentParser(); ap.add_argument("screen"); ap.add_argument("out")
ap.add_argument("--data", default="data/raw/math500_test.jsonl"); ap.add_argument("--easy", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
hits = collections.defaultdict(list)
for l in open(os.path.join(a.screen, "records.jsonl")):
    r = json.loads(l)
    if r["cond"] == "single" and not r.get("error"): hits[r["id"]].append(r["correct"])
k = max(len(v) for v in hits.values())
full = {i: v for i, v in hits.items() if len(v) == k}
contested = sorted(i for i, v in full.items() if sum(v) < k)
easy = sorted(i for i, v in full.items() if sum(v) == k)
pick_easy = sorted(random.Random(a.seed).sample(easy, min(a.easy, len(easy))))
keep = set(contested) | set(pick_easy)
dist = collections.Counter(sum(v) for v in full.values())
print(f"{len(full)} screened problems x {k} seeds; solved-count distribution {dict(sorted(dist.items()))}")
print(f"keeping {len(contested)} contested + {len(pick_easy)} of {len(easy)} easy = {len(keep)}")
with open(a.out, "w") as f:
    for l in open(a.data):
        p = json.loads(l)
        if p["unique_id"] in keep:
            p["stratum"] = "contested" if p["unique_id"] in contested else "easy"
            p["screen_solved"] = sum(full[p["unique_id"]]); f.write(json.dumps(p) + "\n")
