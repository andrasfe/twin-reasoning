#!/usr/bin/env python3
"""Summarize results/<run>/records.jsonl produced by twin.py.

Seeds are tracked per condition (the main run evaluates `single` on fresh seeds, the two-trace conditions on
their own seeds). A problem enters the comparison only when it is complete in every condition.

Per condition: accuracy per seed, mean +- sd, pass@seeds, majority over seeds (sc@k), token / round / forced
statistics. For two-trace conditions (x_ peer-framed, s_ self-framed): per-instance accuracy (each trace judged
alone), pair accuracy (agree, else the instance that finished on its own, else A), agreement, pass@2.
The compute-matched control for the pair readout is `single-pair`: two independent single seeds under the same
readout rule, averaged over seed pairs. Paired bootstrap CIs over problems.
"""
import argparse, collections, itertools, json, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__))
from twin import is_equiv, pair_pick

TWO = ("x_", "s_")

def majority(preds):
    groups = []
    for p in preds:
        if p is None: continue
        for g in groups:
            if is_equiv(p, g[0]): g[1] += 1; break
        else: groups.append([p, 1])
    if not groups: return None
    groups.sort(key=lambda g: -g[1]); return groups[0][0]

def bootstrap_ci(vals, n=4000, seed=0):
    rng = random.Random(seed); m = len(vals)
    if m == 0: return (float("nan"),) * 3
    means = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(vals) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]

def fmt_ci(vals):
    m, lo, hi = bootstrap_ci(vals)
    w = sum(v > 0 for v in vals); l = sum(v < 0 for v in vals)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]  better/worse/same {w}/{l}/{len(vals)-w-l}  (n={len(vals)})"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run")
    ap.add_argument("--all", action="store_true",
                    help="use every problem complete in a condition instead of only problems complete in all conditions")
    ap.add_argument("--stratum", help="only problems of this stratum (contested / easy, set by select_problems.py)")
    ap.add_argument("--seeds", help="comma list: only these seeds")
    ap.add_argument("--conds", help="comma list: only these conditions")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.run, "records.jsonl"))]
    n_err = sum(bool(r.get("error")) for r in recs)
    recs = [r for r in recs if not r.get("error")]
    if a.stratum: recs = [r for r in recs if r.get("stratum") == a.stratum]
    if a.seeds: recs = [r for r in recs if r["seed"] in {int(x) for x in a.seeds.split(",")}]
    if a.conds: recs = [r for r in recs if r["cond"] in a.conds.split(",")]
    conds = sorted({r["cond"] for r in recs}, key=lambda c: (c != "single", c[:2], c))
    by = collections.defaultdict(dict)
    for r in recs: by[(r["cond"], r["id"])][r["seed"]] = r
    seeds = {c: sorted({r["seed"] for r in recs if r["cond"] == c}) for c in conds}
    ids = sorted({r["id"] for r in recs})
    def complete(c, i): return len(by[(c, i)]) == len(seeds[c])
    full = [i for i in ids if all(complete(c, i) for c in conds)]
    print(f"run {a.run}: {len(recs)} records ({n_err} errored, excluded), {len(ids)} problems, "
          f"{len(full)} complete in all {len(conds)} conds; seeds {seeds}")
    def probs(c): return [i for i in ids if complete(c, i)] if a.all else full
    gold = {r["id"]: r["gold"] for r in recs}

    print("\n| cond | acc per seed | mean +- sd | pass@seeds | sc@seeds | gen tok | prompt tok | rounds | forced | wall s | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        P = probs(c)
        if not P: continue
        accs = [sum(by[(c, i)][s]["correct"] for i in P) / len(P) for s in seeds[c]]
        rs = [r for i in P for r in by[(c, i)].values()]
        passk = sum(any(r["correct"] for r in by[(c, i)].values()) for i in P) / len(P)
        sck = sum(is_equiv(majority([r["pred"] for r in by[(c, i)].values()]), gold[i]) for i in P) / len(P)
        sd = st.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"| {c} | {' '.join(f'{x:.3f}' for x in accs)} | {st.mean(accs):.3f} +- {sd:.3f} | {passk:.3f} | {sck:.3f} | "
              f"{st.mean(r['gen_tokens'] for r in rs):.0f} | {st.mean(r['prompt_tokens'] for r in rs):.0f} | "
              f"{st.mean(r['turns'] for r in rs):.1f} | {st.mean(r['forced'] for r in rs):.2f} | {st.mean(r['wall'] for r in rs):.0f} | {len(P)} |")

    # per-problem scores used by every comparison below
    def inst(c, i):   # per-instance accuracy, mean over seeds (single: its own accuracy)
        if c.startswith(TWO): return st.mean((by[(c, i)][s]["correct_a"] + by[(c, i)][s]["correct_b"]) / 2 for s in seeds[c])
        return st.mean(by[(c, i)][s]["correct"] for s in seeds[c])
    def pair(c, i):   # pair readout; for single, two independent seeds under the same rule
        if c.startswith(TWO): return st.mean(by[(c, i)][s]["correct"] for s in seeds[c])
        prs = list(itertools.combinations(seeds[c], 2))
        return st.mean(is_equiv(pair_pick(by[(c, i)][s]["pred"], by[(c, i)][t]["pred"],
                                          by[(c, i)][s]["forced"], by[(c, i)][t]["forced"])[0], gold[i]) for s, t in prs)

    xconds = [c for c in conds if c.startswith(TWO)]
    if xconds:
        print("\nTwo-answer readouts (pair = agree, else the instance that finished on its own, else A):")
        print("| cond | per-instance acc | pair acc | agree | pass@2 | gen tok/pair | stalls | leaks/inserts | n |")
        print("|---|---|---|---|---|---|---|---|---|")
        P = probs("single") if "single" in conds else []
        if P and len(seeds["single"]) >= 2:
            ag = p2 = gen = cnt = 0
            for i in P:
                for s, t in itertools.combinations(seeds["single"], 2):
                    ra, rb = by[("single", i)][s], by[("single", i)][t]
                    ag += pair_pick(ra["pred"], rb["pred"], ra["forced"], rb["forced"])[1]
                    p2 += (ra["correct"] or rb["correct"]); gen += ra["gen_tokens"] + rb["gen_tokens"]; cnt += 1
            print(f"| single-pair (indep.) | {st.mean(inst('single', i) for i in P):.3f} | {st.mean(pair('single', i) for i in P):.3f} | "
                  f"{ag/cnt:.3f} | {p2/cnt:.3f} | {gen/cnt:.0f} | | | {len(P)} |")
        for c in xconds:
            P = probs(c)
            if not P: continue
            rs = [r for i in P for r in by[(c, i)].values()]
            lk = f"{sum(r.get('leaks', 0) for r in rs)}/{sum(r['inserts'] for r in rs)}" if c.startswith("s_") else ""
            print(f"| {c} | {st.mean(inst(c, i) for i in P):.3f} | {st.mean(pair(c, i) for i in P):.3f} | "
                  f"{st.mean(r['agree'] for r in rs):.3f} | {st.mean(r['correct_a'] or r['correct_b'] for r in rs):.3f} | "
                  f"{st.mean(r['gen_tokens'] for r in rs):.0f} | {sum(r.get('stalls', 0) for r in rs)} | {lk} | {len(P)} |")

    def contrast(label, c1, c2, f):
        if c2 == "single" and f is pair and len(seeds["single"]) < 2: return
        P = [i for i in ids if complete(c1, i) and complete(c2, i)] if a.all else full
        if P: print(f"  {label:52s} {fmt_ci([f(c1, i) - f(c2, i) for i in P])}")
    print("\nPaired differences over problems (mean over seeds per problem), bootstrap 95% CI:")
    if "single" in conds:
        for c in conds:
            if c != "single" and not c.startswith(TWO): contrast(f"{c} - single", c, "single", inst)
        for c in xconds:
            contrast(f"{c} per-instance - single", c, "single", inst)
            contrast(f"{c} pair - single-pair (indep.)", c, "single", pair)
    if "x_none" in conds:
        for c in xconds:
            if c != "x_none": contrast(f"{c} per-instance - x_none", c, "x_none", inst)
    def sched(c): return c.partition("@")[2]   # exchange schedule suffix: s_challenge@4 -> "4"
    for c in xconds:
        ref = "s_wait" + ("@" + sched(c) if sched(c) else "")
        if c.startswith("s_") and c != ref and ref in conds: contrast(f"{c} per-instance - {ref}", c, ref, inst)
    for c in xconds:                         # sparse vs dense exchange, same mode and framing
        base = c.partition("@")[0]
        if sched(c) and base in conds:
            contrast(f"schedule {c} - {base} per-instance", c, base, inst)
            contrast(f"schedule {c} - {base} pair", c, base, pair)
    for k in sorted({sched(c) for c in xconds}):
        sfx = "@" + k if k else ""
        for sc, xc in [("s_summary", "x_summary"), ("s_raw", "x_raw"), ("s_challenge", "x_critique"), ("s_wait", "x_none")]:
            sc, xc = sc + sfx, xc + ("" if xc == "x_none" else sfx)
            if sc in conds and xc in conds:   # same information, first-person vs [PEER] framing
                contrast(f"framing {sc} - {xc} per-instance", sc, xc, inst)
                contrast(f"framing {sc} - {xc} pair", sc, xc, pair)
    for c in conds:
        rs = [r for i in probs(c) for r in by[(c, i)].values()]
        if rs: print(f"{c}: no boxed answer in {sum(r['pred'] is None for r in rs)}/{len(rs)} runs")

if __name__ == "__main__":
    main()
