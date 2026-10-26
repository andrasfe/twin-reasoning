#!/usr/bin/env python3
"""Twin reasoning pilot harness.

ONE small local model served by LM Studio (OpenAI-compatible /v1/completions) is run as two
"instances" that differ only in context and sampling.  Conditions:

  single        one instance, one trace (baseline at per-trace budget T)

  x_<mode>      CROSS-POLLINATION (primary design): two instances reason IN PARALLEL, each in its
                own trace.  After every chunk of reasoning both are paused, material from the peer
                is pasted into each trace between [PEER] ... [/PEER] markers, and both continue
                their OWN line of reasoning.  <mode> says what is pasted:
                  raw       the peer's latest chunk, verbatim
                  summary   the peer's own <=3-sentence summary of its reasoning so far (extra call)
                  answer    the peer's current candidate answer + one-sentence reason (extra call)
                  critique  the peer's <=3-sentence check of MY latest reasoning (extra call)
                  none      no exchange at all: identical mechanics, no information flow (control)
                Each instance answers; the pair answer is the agreed answer, else the answer of the
                instance that finished on its own, else A.  The compute-matched control for the
                pair readout is two independent `single` seeds under the same rule (summarize.py).

  twin_asym /   SHARED-TRACE TURN-TAKING (secondary design): two instances alternate paragraph by
  twin_sym      paragraph on one trace, with solver/checker roles or identical roles.

Modes: native thinking (<think> block, default) or --nothink (reasoning in the visible answer).
Raw ChatML prompts are built here so the assistant prefix is under our control.
"""
import argparse, json, os, random, re, subprocess, sys, threading, time
import concurrent.futures as cf
import httpx

BASE = os.environ.get("LMS_BASE", "http://localhost:1234/v1")
MODEL = os.environ.get("LMS_MODEL", "twin")
LOAD_CMD = os.environ.get("LMS_LOAD_CMD", os.path.expanduser("~/.lmstudio/bin/lms")
                          + " load qwen3.6-35b-a3b-mlx --identifier twin -c 16384 --parallel 2 --ttl 360000 -y")
MAX_INFLIGHT = int(os.environ.get("LMS_PARALLEL", "2"))

# ----------------------------------------------------------------------------- LLM
# the slowest legitimate request (a 3000-token single run) takes ~1-2 min; a hung request must not stall a run for hours
_client = httpx.Client(timeout=httpx.Timeout(float(os.environ.get("LMS_TIMEOUT", "300")), connect=10.0))
_inflight = threading.Semaphore(MAX_INFLIGHT)
_reload_lock = threading.Lock()
_last_reload = [0.0]

def _ensure_loaded(hard=False):
    """Watchdog (at most once per 60 s). The MLX engine has crashed mid-run ("No models loaded") and has also
    wedged with requests stuck in prompt processing; hard=True unloads the wedged instance before reloading."""
    with _reload_lock:
        if time.time() - _last_reload[0] < 60: return
        cmd = (os.path.expanduser("~/.lmstudio/bin/lms") + f" unload {MODEL}; " if hard else "") + LOAD_CMD
        print(f"[watchdog {time.strftime('%H:%M:%S')}] {'engine wedged' if hard else 'model not loaded'}; running: {cmd}",
              file=sys.stderr, flush=True)
        try: subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        except Exception as e: print("[watchdog] reload failed:", e, file=sys.stderr, flush=True)
        _last_reload[0] = time.time()

def complete(prompt, max_tokens, temperature, stop=None, top_p=0.95, seed=None):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p}
    if stop: payload["stop"] = stop
    if seed is not None: payload["seed"] = seed
    last = None
    for attempt in range(8):
        try:
            with _inflight:
                r = _client.post(BASE + "/completions", json=payload)
            if r.status_code == 200:
                j = r.json(); ch = j["choices"][0]; u = j.get("usage", {})
                return ch["text"], ch.get("finish_reason"), u.get("completion_tokens", 0), u.get("prompt_tokens", 0)
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            if "No models loaded" in r.text: _ensure_loaded(); time.sleep(5); continue
            if "has crashed" in r.text or "fatal exception" in r.text: _ensure_loaded(hard=True); time.sleep(5); continue
        except httpx.TimeoutException as e:
            last = e; _ensure_loaded(hard=True)
        except Exception as e:
            last = e
        print(f"[retry {time.strftime('%H:%M:%S')} attempt {attempt + 1}] {str(last)[:160]}", file=sys.stderr, flush=True)
        time.sleep(2 * (attempt + 1))
    raise last

# ----------------------------------------------------------------------------- prompts
USER_SUFFIX = "\n\nSolve the problem. Put the final answer in \\boxed{}."
SYS_SINGLE = "You are a careful, expert mathematician."
END_THINK = "writing </think> and then the final answer"
END_NOTHINK = "giving the final answer in \\boxed{}"

X_PREAMBLE = (
    "You are one of TWO instances of the same model solving the same problem at the same time, each in "
    "your OWN scratchpad. Every paragraph you write starts with the marker [ME]. {peer_sentence}Continue "
    "YOUR OWN reasoning from where you left off; never restart from scratch. When you are confident, "
    "finish by {ending}."
)
PEER_OPEN, PEER_CLOSE, ME_TAG = "[PEER]", "[/PEER]", "[ME]"
X_PEER = ("At intervals, material from the other instance is pasted into your scratchpad between [PEER] "
          "and [/PEER] markers: {what}. It is the peer's work, not yours, and it may be right or wrong. "
          "Use it as you see fit: adopt it, check it, or ignore it. ")
X_WHAT = {"raw": "its most recent reasoning, verbatim",
          "summary": "its own short summary of its reasoning so far",
          "answer": "its current candidate answer with a one-sentence reason",
          "critique": "its short check of your reasoning, informed by its own"}
# A cue ends on the opening words of the insert (its "lead"); the lead is kept in the inserted text.
SUMMARY_CUE, SUMMARY_LEAD = ("\n\n[Summary of my reasoning so far for my peer, at most 3 sentences, including any candidate "
                             "answer:] In short,"), "In short,"
ANSWER_CUE, ANSWER_LEAD = "\n\n[My current candidate answer and a one-sentence reason, for my peer:] My current answer is", "My current answer is"
NOTE_PEER = "\n\n[The other instance's own reasoning so far:\n<<<\n{peer}\n>>>]"
CRITIQUE_CUE = ("\n\n[Check of MY reasoning above, written by the other instance in at most 3 sentences, drawing on "
                "its own reasoning: point out where we disagree or any error, recomputing the key step, or confirm it:] Check:")
CRITIQUE_LEAD = "Check:"

# Self framing (s_ conditions): the model is never told that a second instance exists. Material derived from
# the peer is woven into its scratchpad in the first person, as its own doubt or second line of attack.
SELF_WAIT = "Wait, let me challenge what I have so far: is any step above wrong, or is there a better approach?"
SELF_RAW_LEAD = "Hmm, let me look at this from another angle.\n\n"
SELF_SUMMARY_CUE = "\n\nWait, let me quickly recap what I have concluded so far, in at most 3 sentences. In short,"
SELF_SUMMARY_LEAD = "Wait, let me consider a different line of reasoning:"
NOTE_OTHER = ("\n\n[Private note, not part of my scratchpad: an independent attempt at this problem reasoned as "
              "follows.\n<<<\n{peer}\n>>>\n")
SELF_CHALLENGE_CUE = ("Write my next thought in the first person, as a challenge to my own reasoning above, in at most 3 "
                      "sentences: where that attempt disagrees with mine or checks something I have not, question my "
                      "step and say what to recompute; if it agrees, question my weakest step instead. Never mention "
                      "the attempt or this note.]\n\nWait,")
SELF_CHALLENGE_LEAD = "Wait,"
LEAK_RE = re.compile(r"(?i)\b(peer|independent attempt|other attempt|the attempt|that attempt|private note|"
                     r"this note|the note|other instance|another instance)\b")
MODES = {"peer": ("none", "raw", "summary", "answer", "critique"), "self": ("wait", "raw", "summary", "challenge")}

TWIN_PREAMBLE = (
    "You are one of TWO instances of the same model that reason TOGETHER in ONE shared "
    "scratchpad. You take turns, one paragraph at a time; the paragraphs are tagged "
    "[A] and [B]. Read everything already written, then continue the SAME line of "
    "reasoning from where it stands. Never restart from scratch, never repeat what is "
    "already established, never address the other instance by name. "
    "When the solution is complete and verified, the instance whose turn it is ends the "
    "scratchpad by {ending}."
)
ROLE_SOLVER = (" You are instance {name}, the SOLVER: drive the solution forward with the "
               "next concrete step and carry out the computations.")
ROLE_CHECKER = (" You are instance {name}, the CHECKER: re-derive and verify what the previous "
                "paragraphs claim, recomputing key quantities independently. If you find an "
                "error, state the correction precisely. If everything holds, say so in one "
                "sentence and state what should be done next.")
ROLE_PLAIN = " You are instance {name}. Continue the reasoning."

def system_for(cond, name, think=True):
    ending = END_THINK if think else END_NOTHINK
    if cond == "single" or cond.startswith("s_"): return SYS_SINGLE
    if cond.startswith("x_"):
        mode = cond[2:]
        peer = "" if mode == "none" else X_PEER.format(what=X_WHAT[mode])
        return X_PREAMBLE.format(peer_sentence=peer, ending=ending)
    pre = TWIN_PREAMBLE.format(ending=ending)
    if cond == "twin_asym":
        return pre + (ROLE_SOLVER if name == "A" else ROLE_CHECKER).format(name=name)
    if cond == "twin_sym":
        return pre + ROLE_PLAIN.format(name=name)
    raise ValueError(cond)

def build_prompt(system, problem, trace, think=True):
    head = "<think>\n" if think else "<think>\n\n</think>\n\n"
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{problem}{USER_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n{head}{trace}")

BUDGET_SLACK = 16   # a trace within this many tokens of its budget counts as out of budget
FORCE_STOP = "\n\nI must stop thinking here and commit to the best answer I have.\n"
FORCE_BOX = "\n\nTherefore, the final answer is $\\boxed{"
FORCE_COMMIT = "\n\nI must stop here and commit to the best answer I have."   # no-think: the trace may end mid-formula

# ----------------------------------------------------------------------------- answer checking
def last_boxed(text):
    if not text: return None
    i = text.rfind("\\boxed")
    if i < 0: return None
    j = text.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0: return text[j+1:k]
    return None

def _strip_text(s):
    return re.sub(r"\\(?:text|mathrm|textbf|mbox)\s*\{([^{}]*)\}", r"\1", s)

def normalize(s):
    if s is None: return None
    s = s.strip()
    s = re.sub(r"(\d)\s*\\(?:text|mbox|mathrm)\s*\{\s*[A-Za-z][A-Za-z ]*\}$", r"\1", s)   # trailing unit: 5.4 \text{ cents}
    s = _strip_text(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\displaystyle", "")
    s = s.rstrip(".")
    s = re.sub(r"\s+", "", s)
    if s.count("=") == 1 and not s.startswith("="):
        s = s.split("=")[1]
    s = s.replace("{,}", ",")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s): s = s.replace(",", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    return s

def to_number(s):
    if s is None: return None
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try: return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        except ZeroDivisionError: return None
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m: return int(m.group(1)) / int(m.group(2))
    try: return float(s)
    except ValueError: return None

INEQ_RE = re.compile(r"(.+?)(<|\\leq?|\\lt)(\\?[A-Za-z]+)(<|\\leq?|\\lt)(.+)")

def is_equiv(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None: return False
    if p == g: return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None: return abs(pn - gn) <= 1e-6 * max(1, abs(gn))
    m = INEQ_RE.fullmatch(p)   # 3<\lambda\le4 for an interval gold (3,4]
    if m and g[:1] in "([" and g[-1:] in ")]":
        p = ("(" if m.group(2) == "<" else "[") + m.group(1) + "," + m.group(5) + (")" if m.group(4) == "<" else "]")
        if p == g: return True
    if g.startswith("(") and g.endswith(")") and "," in p and p[:1] not in "([\\": p = "(" + p + ")"   # 6,31,-1 for (6,31,-1)
    if (p.startswith("(") and g.startswith("(")) or (p.startswith("\\{") and g.startswith("\\{")):
        pe = p.strip("()\\{}").split(","); ge = g.strip("()\\{}").split(",")
        if len(pe) == len(ge) and all(is_equiv(a, b) for a, b in zip(pe, ge)): return True
    return False

def pair_pick(pred_a, pred_b, forced_a, forced_b):
    """Deterministic two-answer readout: agreement, else the instance that finished on its own, else A."""
    if pred_a is not None and pred_b is not None and is_equiv(pred_a, pred_b): return pred_a, True
    if pred_a is None: return pred_b, False
    if pred_b is None: return pred_a, False
    if forced_a != forced_b: return (pred_b if forced_a else pred_a), False
    return pred_a, False

# ----------------------------------------------------------------------------- building blocks
def gen_chunk(system, problem, trace, tag, temp, budget_left, args, seed, extra_stop=()):
    """Generate one chunk (>= step_min tokens unless the model finishes, <= step_max, ending at a
    paragraph boundary) continuing `trace` after the forced `tag`. `extra_stop` are markers the
    model must not produce (e.g. the peer markers). Returns (text, tokens, finished, prompt_tokens)."""
    step = ""; step_tokens = 0; ptoks = 0; finished = False; empties = 0
    while True:
        remaining = budget_left - step_tokens
        if remaining < BUDGET_SLACK: break
        prefix = trace + tag + step
        text, fr, ct, pt = complete(build_prompt(system, problem, prefix, think=args.think),
                                    min(args.step_max, remaining), temp, stop=["\n\n", *extra_stop], seed=seed)
        for m in extra_stop: text = text.split(m)[0]
        step += text; step_tokens += ct; ptoks += pt
        if not text.strip():
            empties += 1
            if empties >= 2: break      # model keeps trying to emit a forbidden marker: end the chunk
        else: empties = 0
        if args.think and "</think>" in step:
            step = step.split("</think>")[0]; finished = True; break
        if not args.think and last_boxed(step) is not None:
            finished = True; break
        if fr == "length" and ct >= min(args.step_max, remaining):
            if step_tokens >= args.step_max: break   # long paragraph: hand over mid-paragraph
            continue
        if step_tokens >= args.step_min: break
        step += "\n\n"                                # short paragraph: keep going
    return step, step_tokens, finished, ptoks

def finalize(system, problem, trace, forced, args, seed, own=None):
    """Visible answer. Think mode: close the block and let the model answer; no-think: the trace is
    the answer. If no \\boxed{} is present, force one with a fixed suffix (same for every condition)."""
    gen = ptoks = 0
    if args.think:
        close = (FORCE_STOP if forced else "\n") + "</think>\n\n"
        base = build_prompt(system, problem, trace.rstrip("\n") + close)
        text, fr, ct, pt = complete(base, args.answer_tokens, args.temp_answer, seed=seed)
        gen += ct; ptoks += pt
    else:
        base = build_prompt(system, problem, "", think=False)
        text = trace
    if last_boxed(text if (args.think or own is None) else own) is None:
        suffix = ("" if args.think else FORCE_COMMIT) + FORCE_BOX
        # one line only: unconstrained, the model has escaped the box with a blank line and kept reasoning
        t2, fr2, ct2, pt2 = complete(base + text.rstrip("\n") + suffix, 60, 0.0, stop=["\n"], seed=seed)
        text = text.rstrip("\n") + suffix + t2
        gen += ct2; ptoks += pt2
    return text, gen, ptoks

# ----------------------------------------------------------------------------- conditions
def run_single(problem, args, seed):
    system = system_for("single", "A")
    t0 = time.time()
    text, fr, ct, pt = complete(build_prompt(system, problem, "", think=args.think), args.think_tokens,
                                args.temp_a, stop=["</think>"] if args.think else None, seed=seed)
    forced = (fr == "length") and (args.think or last_boxed(text) is None)
    ans, ct2, pt2 = finalize(system, problem, text, forced, args, seed)
    return dict(trace=text, answer_text=ans, gen_tokens=ct + ct2, prompt_tokens=pt + pt2,
                forced=forced, turns=1, wall=time.time() - t0)

def run_twin_shared(problem, args, seed, cond):
    names = ["A", "B"]
    systems = {n: system_for(cond, n, args.think) for n in names}
    temps = {"A": args.temp_a, "B": args.temp_b}
    trace = ""; gen = 0; ptoks = 0; turn = 0; forced = False; done = False
    t0 = time.time(); turns_log = []
    while True:
        name = names[turn % 2]
        step, st, done, pt = gen_chunk(systems[name], problem, trace, f"[{name}]" if args.tags else "",
                                       temps[name], args.think_tokens - gen, args, None if seed is None else seed * 1000 + turn)
        gen += st; ptoks += pt
        trace = trace + (f"[{name}]" if args.tags else "") + step.rstrip("\n") + "\n\n"
        turns_log.append((name, st)); turn += 1
        if done: break
        if gen >= args.think_tokens - BUDGET_SLACK or turn >= args.max_turns or st == 0: forced = True; break
    finisher = names[(turn - 1) % 2] if done else names[turn % 2]
    ans, ct2, pt2 = finalize(systems[finisher], problem, trace, forced, args, seed)
    return dict(trace=trace, answer_text=ans, gen_tokens=gen + ct2, prompt_tokens=ptoks + pt2,
                forced=forced, turns=turn, turns_log=turns_log, finisher=finisher, wall=time.time() - t0)

def run_twin_x(problem, args, seed, mode, frame="peer", every=None):
    """Two parallel traces with periodic inserts. frame="peer": the model knows about the other instance and
    inserts are marked [PEER]...[/PEER]. frame="self": the model believes it is alone and inserts are written
    in the first person, as its own self-challenge. Exchanges happen after every `every` rounds
    (default --exchange-every); a condition name like s_challenge@4 sets it per condition."""
    every = every or args.exchange_every
    if mode not in MODES[frame]: raise ValueError(f"mode {mode} not available for frame {frame}")
    names = ["A", "B"]; other = {"A": "B", "B": "A"}
    selfx = frame == "self"
    system = SYS_SINGLE if selfx else system_for("x_" + mode, "A", args.think)
    temps = {"A": args.temp_a, "B": args.temp_b}
    own = {n: "" for n in names}      # the instance's own text only
    trace = {n: "" for n in names}    # own text + [PEER] inserts, i.e. what the instance sees
    gen = {n: 0 for n in names}; done = {}; forced = {n: False for n in names}
    ptoks = 0; xgen = 0; rounds = 0; inserts = 0; leaks = 0; stalls = 0; log = []
    t_chunk = t_insert = 0.0          # wall time spent in the chunk phase vs the exchange phase
    t0 = time.time()
    pool = cf.ThreadPoolExecutor(2)
    tag = ME_TAG if (args.tags and not selfx) else ""
    xstop = () if selfx else (PEER_OPEN, PEER_CLOSE)
    fresh = set()                        # instances that wrote new material since the last exchange
    since = {n: "" for n in names}       # ... and that material
    while len(done) < 2:
        active = [n for n in names if n not in done]
        fresh |= set(active)
        tc = time.time()
        futs = {n: pool.submit(gen_chunk, system, problem, trace[n], tag, temps[n], args.think_tokens - gen[n], args,
                               None if seed is None else seed * 1000 + rounds * 2 + names.index(n),
                               xstop) for n in active}
        for n in active:
            step, st, fin, pt = futs[n].result()
            own[n] += step.rstrip("\n") + "\n\n"; trace[n] += tag + step.rstrip("\n") + "\n\n"
            since[n] += step.rstrip("\n") + "\n\n"; gen[n] += st; ptoks += pt
            if fin: done[n] = "self"
            elif gen[n] >= args.think_tokens - BUDGET_SLACK: done[n] = "budget"; forced[n] = True
            elif st == 0:   # produced nothing: stop instead of looping on empty chunks (this once flooded the server)
                done[n] = "stall"; forced[n] = True; stalls += 1
                print(f"[stall] instance {n} produced no tokens in round {rounds}", file=sys.stderr, flush=True)
        rounds += 1; t_chunk += time.time() - tc
        if rounds >= args.max_turns:
            for n in names:
                if n not in done: done[n] = "rounds"; forced[n] = True
        active = [n for n in names if n not in done]
        if mode == "none" or not active or rounds % every: continue

        def make_insert(n):
            p = other[n]
            if mode == "wait": return SELF_WAIT, 0, 0          # fixed self-challenge, no peer information
            if p not in fresh or not since[p].strip(): return "", 0, 0   # a finished peer sends one last note, then goes quiet
            if mode == "raw": return ((SELF_RAW_LEAD + since[p].strip()) if selfx else since[p].strip()), 0, 0
            if mode == "summary":
                ctx, cue = own[p], (SELF_SUMMARY_CUE if selfx else SUMMARY_CUE)
                lead = SELF_SUMMARY_LEAD if selfx else SUMMARY_LEAD
            elif mode == "answer": ctx, cue, lead = own[p], ANSWER_CUE, ANSWER_LEAD
            elif mode in ("critique", "challenge"):             # written on MY text, informed by the peer's own text
                note = (NOTE_OTHER if selfx else NOTE_PEER).format(peer=own[p].strip()[-12000:])
                ctx, cue = own[n], note + (SELF_CHALLENGE_CUE if selfx else CRITIQUE_CUE)
                lead = SELF_CHALLENGE_LEAD if selfx else CRITIQUE_LEAD
            else: raise ValueError(mode)
            text, fr, ct, pt = complete(build_prompt(system, problem, ctx.rstrip("\n") + cue, think=args.think),
                                        args.insert_tokens, 0.3,
                                        stop=["\n\n", "</think>", PEER_OPEN, PEER_CLOSE, ME_TAG.strip(), "[Private", "<<<"],
                                        seed=None if seed is None else seed * 1000 + 500 + rounds)
            text = text.rstrip().lstrip("\n")
            if not text.strip(): return "", ct, pt
            return lead + (text if text[0] in " ,.;:" else " " + text), ct, pt
        ti = time.time()
        futs = {n: pool.submit(make_insert, n) for n in active}
        for n in active:
            txt, ct, pt = futs[n].result(); xgen += ct; ptoks += pt
            if txt:
                trace[n] += (f"{txt}\n\n" if selfx else f"{PEER_OPEN} {txt} {PEER_CLOSE}\n\n"); inserts += 1
                if selfx and mode != "wait" and LEAK_RE.search(txt): leaks += 1
                log.append((rounds, n, len(txt)))
        t_insert += time.time() - ti
        fresh = set(); since = {n: "" for n in names}
    futs = {n: pool.submit(finalize, system, problem, trace[n], forced[n], args, seed, own[n]) for n in names}
    ans = {n: futs[n].result() for n in names}
    pool.shutdown()
    preds = {n: last_boxed(ans[n][0]) for n in names}
    pick, agree = pair_pick(preds["A"], preds["B"], forced["A"], forced["B"])
    return dict(trace=trace["A"], trace_b=trace["B"], answer_text=ans["A"][0], answer_text_b=ans["B"][0],
                pred_a=preds["A"], pred_b=preds["B"], pair_pred=pick, agree=agree,
                done_a=done["A"], done_b=done["B"], gen_a=gen["A"], gen_b=gen["B"], exchange_tokens=xgen,
                gen_tokens=gen["A"] + gen["B"] + xgen + ans["A"][1] + ans["B"][1],
                prompt_tokens=ptoks + ans["A"][2] + ans["B"][2],
                forced=forced["A"] and forced["B"], turns=rounds, inserts=inserts, leaks=leaks, stalls=stalls, insert_log=log,
                t_chunk=t_chunk, t_insert=t_insert,
                wall=time.time() - t0)

# ----------------------------------------------------------------------------- driver
def load_problems(path, level, n, shuffle_seed):
    rows = [json.loads(l) for l in open(path)]
    if level: rows = [r for r in rows if r["level"] in level]
    random.Random(shuffle_seed).shuffle(rows)
    return rows[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/math500_test.jsonl")
    ap.add_argument("--level", default="5", help="comma list of MATH levels, '' for all")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--conds", default="single,x_none,x_raw,x_summary")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--think-tokens", type=int, default=3000, help="per-trace generation budget T")
    ap.add_argument("--answer-tokens", type=int, default=1000)
    ap.add_argument("--nothink", dest="think", action="store_false",
                    help="reason in the visible answer instead of the native <think> block")
    ap.add_argument("--step-min", type=int, default=150, help="min tokens per chunk/turn")
    ap.add_argument("--step-max", type=int, default=350, help="max tokens per chunk/turn")
    ap.add_argument("--max-turns", type=int, default=60, help="max turns (shared) / rounds (x_)")
    ap.add_argument("--exchange-every", type=int, default=1, help="x_: exchange after every k rounds")
    ap.add_argument("--insert-tokens", type=int, default=150, help="x_: max tokens of a generated insert")
    ap.add_argument("--temp-a", type=float, default=0.7)
    ap.add_argument("--temp-b", type=float, default=0.7)
    ap.add_argument("--temp-answer", type=float, default=0.3)
    ap.add_argument("--no-tags", dest="tags", action="store_false")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    level = [int(x) for x in args.level.split(",")] if args.level else None
    probs = load_problems(args.data, level, args.n, args.shuffle_seed)
    conds = args.conds.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.jsonl"), "a") as f:   # one line per invocation (runs are resumed/extended)
        f.write(json.dumps(dict(vars(args), started=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    rec_path = os.path.join(args.out, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        for l in open(rec_path):
            r = json.loads(l)
            if not r.get("error"): done.add((r["id"], r["cond"], r["seed"]))
    jobs = [(p, c, s) for p in probs for c in conds for s in seeds if (p["unique_id"], c, s) not in done]
    print(f"{len(probs)} problems x {conds} x seeds {seeds}: {len(jobs)} jobs to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def work(job):
        p, c, s = job
        try:
            base, _, k = c.partition("@")      # x_critique@4: exchange after every 4th round
            if c == "single": res = run_single(p["problem"], args, s)
            elif c.startswith("x_"): res = run_twin_x(p["problem"], args, s, base[2:], "peer", int(k) if k else None)
            elif c.startswith("s_"): res = run_twin_x(p["problem"], args, s, base[2:], "self", int(k) if k else None)
            else: res = run_twin_shared(p["problem"], args, s, c)
        except Exception as e:
            res = dict(error=repr(e), trace="", answer_text="", gen_tokens=0, prompt_tokens=0, forced=True, turns=0, wall=0)
        pred = res.get("pair_pred") if "pair_pred" in res else last_boxed(res["answer_text"])
        rec = dict(id=p["unique_id"], level=p["level"], subject=p["subject"], stratum=p.get("stratum"), cond=c, seed=s,
                   gold=p["answer"], pred=pred, correct=is_equiv(pred, p["answer"]), **res)
        if "pred_a" in res:
            rec["correct_a"] = is_equiv(res["pred_a"], p["answer"]); rec["correct_b"] = is_equiv(res["pred_b"], p["answer"])
        with lock:
            with open(rec_path, "a") as f: f.write(json.dumps(rec) + "\n")
            extra = (f"A={int(rec['correct_a'])} B={int(rec['correct_b'])} agree={int(rec['agree'])} ins={rec['inserts']}"
                     if "pred_a" in res else "")
            print(f"[{c:10s} s{s}] {p['unique_id']:30s} gen={rec['gen_tokens']:5d} prompt={rec['prompt_tokens']:6d} "
                  f"turns={rec['turns']:2d} forced={int(rec['forced'])} {'OK ' if rec['correct'] else 'XX '} "
                  f"pred={str(pred)[:22]!r} gold={p['answer'][:22]!r} {rec['wall']:.0f}s {extra}"
                  f"{' ERROR ' + rec['error'][:80] if rec.get('error') else ''}", flush=True)
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))
    print("done", flush=True)

if __name__ == "__main__":
    main()
