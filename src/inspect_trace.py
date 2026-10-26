#!/usr/bin/env python3
"""Print traces from a run: python src/inspect_trace.py results/<run> [--cond twin_asym] [--id substr] [--wrong]"""
import argparse, json, os

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--cond"); ap.add_argument("--id")
ap.add_argument("--seed", type=int); ap.add_argument("--wrong", action="store_true"); ap.add_argument("--chars", type=int, default=6000)
a = ap.parse_args()
for l in open(os.path.join(a.run, "records.jsonl")):
    r = json.loads(l)
    if a.cond and r["cond"] != a.cond: continue
    if a.id and a.id not in r["id"]: continue
    if a.seed is not None and r["seed"] != a.seed: continue
    if a.wrong and r["correct"]: continue
    print("=" * 100)
    print(f"{r['cond']} seed {r['seed']} {r['id']} L{r['level']} correct={r['correct']} pred={r['pred']!r} gold={r['gold']!r} "
          f"gen={r['gen_tokens']} prompt={r['prompt_tokens']} turns={r['turns']} forced={r['forced']} wall={r['wall']:.0f}s "
          f"{'ERROR ' + r['error'] if r.get('error') else ''}")
    if r.get("turns_log"): print("turns:", " ".join(f"{n}{t}" for n, t in r["turns_log"]), "finisher:", r.get("finisher"))
    if "pred_a" in r:
        print(f"A: pred={r['pred_a']!r} correct={r['correct_a']} done={r['done_a']} gen={r['gen_a']} | "
              f"B: pred={r['pred_b']!r} correct={r['correct_b']} done={r['done_b']} gen={r['gen_b']} | agree={r['agree']} "
              f"inserts={r['inserts']} leaks={r.get('leaks')} exchange_tokens={r.get('exchange_tokens')}")
    print("-" * 40, "trace A" if "trace_b" in r else "trace")
    tr = r["trace"]; print(tr[: a.chars] + ("\n... [%d more chars]" % (len(tr) - a.chars) if len(tr) > a.chars else ""))
    if "trace_b" in r:
        print("-" * 40, "trace B")
        tb = r["trace_b"]; print(tb[: a.chars] + ("\n... [%d more chars]" % (len(tb) - a.chars) if len(tb) > a.chars else ""))
    print("-" * 40, "answer" + (" A" if "trace_b" in r else ""))
    an = r["answer_text"]; print(an[-800:])
    if "answer_text_b" in r:
        print("-" * 40, "answer B"); print(r["answer_text_b"][-800:])
