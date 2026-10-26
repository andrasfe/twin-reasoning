# Twin reasoning: interlacing the reasoning of two instances of the same LLM

Status: pilot design, 2026-10-25. Model under test: `qwen3.6-35b-a3b-mlx` (35B total, ~3B active,
8-bit MLX) served by LM Studio on an M4 Max. Benchmark: MATH-500, level-5 subset.

## 1. The question

Two instances of one small model reason **in parallel, each in its own trace**. At intervals both
are paused, something from each one's reasoning is pasted into the other's trace, and both
**continue their own line of reasoning** (no restart, no turn-taking). Does this cross-pollination
produce better answers than the same model reasoning alone, and than two independent runs of the
same model at the same total cost? And *what* should be exchanged?

Spending a per-trace budget of T tokens, the comparisons are:

| condition | traces | information flow | generated tokens |
|---|---|---|---|
| single | 1 | none | ~T |
| x_none | 2 | none (identical mechanics, control) | ~2T |
| x_raw | 2 | peer's latest chunk, verbatim | ~2T |
| x_summary | 2 | peer's own ≤3-sentence summary of its reasoning so far | ~2T + summaries |
| x_answer | 2 | peer's current candidate answer + one reason | ~2T + small |
| x_critique | 2 | ≤3-sentence check of *my* reasoning, written with the peer's reasoning in view | ~2T + critiques |
| s_wait | 2 | none; a fixed first-person "Wait, let me challenge what I have so far..." | ~2T |
| s_raw | 2 | peer's latest chunk, introduced as "Hmm, let me look at this from another angle." | ~2T |
| s_summary | 2 | peer's summary, introduced as "Wait, let me consider a different line of reasoning:" | ~2T + summaries |
| s_challenge | 2 | first-person "Wait, ..." objection to my reasoning, written with the peer's reasoning in view | ~2T + challenges |
| single-pair | 2 x single | none; two independent seeds, same readout rule | ~2T |
| sc@3 | 3 x single | majority vote | ~3T |

**Two framings.** In `x_` conditions each instance is told a peer exists and receives `[PEER] ... [/PEER]`
blocks. In `s_` conditions the instance is never told: its system prompt is the single-solver prompt, and the
peer-derived material is written into its scratchpad in the first person, as its own doubt or second line of
attack (a self-challenging argument rather than a peer weighing in). `s_summary` vs `x_summary` and
`s_challenge` vs `x_critique` carry the same information and differ only in framing. `s_wait` carries no peer
information at all: it isolates the effect of the self-challenge cue itself (the s1 "Wait" effect).

Two readouts are scored for every two-trace condition: **per-instance** (each trace judged alone, the
clean test of "does seeing the peer help a trace") and **pair** (agree, else the instance that
finished on its own, else A; the clean test of "does the pair beat two independent runs").

## 2. Why this is not obviously a win: the identity argument

Two instances with the same weights, context and sampling parameters are the same stochastic
process. All benefit must come from something that breaks the symmetry. Here the symmetry breaker
is *information flow between two independently sampled traces*: each instance sees a second,
differently sampled attempt at the same problem while it is still reasoning. That can help in two
ways: (a) error correction, when the peer's numbers disagree with mine I re-check; (b) proposal
supply, when the peer has a better approach I can adopt it. It can hurt in two ways: (c) herding,
both converge on the first confident-looking answer (the Debate-or-Vote martingale result); (d)
distraction, context grows with pasted material and small models lose the thread (Hogwild's
small-model failure). `x_none` isolates the mechanics; `single-pair` isolates the readout rule.

## 3. Prior art (searched 2026-10-25)

Nobody appears to have tested exactly this: API-level, chunk-synchronous exchange between two
copies of one small model, with the exchanged *content* varied, against compute-matched controls.
The neighbours set the bar, and it is not encouraging:

- **Hogwild! Inference** (Rodionov et al., arXiv:2504.06261): several copies of one model decode
  *concurrently* against one shared KV cache; each sees the others' partial steps. Beats
  self-consistency on LIMO at matched budget for QwQ-32B and larger Qwen3 models; **Qwen3-1.7B
  and 4B get distracted and fail**; overthinking turned correct answers wrong on physics. Needs a
  custom engine. Our `x_raw` is the API-only, synchronous cousin.
- **Group Think** (arXiv:2505.11107), **LACE** (2604.15529), **LaneRoPE** (2605.27570): token-level
  mutual visibility between parallel threads, all needing modified decoders.
- **Multi-agent debate** (Du et al. 2023) and its critics: *Should we be going MAD?* (Smit et al.,
  ICML 2024): debate does not reliably beat self-consistency, very prompt-sensitive. *Debate or
  Vote* (NeurIPS 2025, arXiv:2508.17536): voting explains most of the gain; debate alone is a
  martingale over beliefs. *Beyond Symmetric Agents* (arXiv:2609.35875, 23 small models): debate
  ties or loses to matched-budget self-consistency at 3.4x tokens; persona prompts reduce accuracy
  ("persona tax"); nearly all benefit comes from the first exchange of answers. *When Debate Helps*
  (arXiv:2610.04686): debate beats voting only with correct proposals in the pool AND a
  verification-aware readout. *Where Reasoning Diverges* (arXiv:2608.01463): exchange only the
  intermediate claims that differ.
- **LeaP, Learning from Peers in Reasoning Models** (Luo et al., arXiv:2505.07787): the closest prior
  work to `x_summary`. N parallel paths; every T = 2-6k tokens each path writes a ≤256-token summary
  triggered in the first person ("Wait, let me quickly recap what I've concluded so far. In short, my
  current conclusions are that ..."), and the top-k most *dissimilar* peer summaries are inserted inside a
  `<peer_summaries>` block followed by "Hmm, it seems that my peers have given me some comments...".
  +4 to +6 points pass@1 on AIME for R1-Distill-7B/14B and QwQ-32B at comparable tokens. Small models
  often fail to follow the summarise/reflect instructions (they fine-tune LeaP-T for this), and one case
  study shows QwQ-32B rejecting the peer framing outright ("But since I'm alone, I'll proceed").
- **Collaborative Parallel Thinking** (arXiv:2605.27030), on Qwen3-30B-A3B-Thinking (our model's
  predecessor): extracted findings go to a deduplicated shared pool broadcast in the input context, not
  injected into the trace; broadcasting starts only after the rate of new discoveries drops. Beats LeaP
  on accuracy vs latency.
- **Can Reasoning Models Detect Changes to their Chains of Thought?** (arXiv:2606.22085): unalerted
  models essentially never notice edits to their own CoT, and a relevant inserted step is detected *below*
  chance. So first-person inserts (`s_`) should be absorbed as the model's own thought.
- **Thinking Intervention** (arXiv:2503.24370): first-person instructions inserted into the thinking
  process steer reasoning models; the mechanism `s_` relies on, here carrying peer content.
- **Self-MoA** (TMLR 2026, arXiv:2502.00674): sampling one model repeatedly and aggregating beats
  mixing different models, supporting the "same model twice" premise.
- **Reasoning Models Generate Societies of Thought** (arXiv:2601.10825): reasoning models already
  simulate several perspectives inside one think block; the marginal value of an external peer may
  be small when native thinking is on. Hence a no-think track as well.

## 4. Protocol (src/twin.py)

Raw ChatML to `/v1/completions` so the assistant prefix (the trace) is under our control. LM Studio
ignores `enable_thinking:false` for this model; a raw `<think>\n` prefix gives native thinking, a
raw `<think>\n\n</think>\n\n` disables it (the `--nothink` track reasons in the visible answer).

Cross-pollination round, both instances in parallel (the server has 2 slots):

1. Each instance generates a **chunk**: continue its trace with a stop at a blank line, at least
   `step_min`=150 tokens (short paragraphs keep going), at most `step_max`=350.
2. If an instance finished (`</think>`, or a `\boxed{}` in no-think mode) it is done; if it hit
   the per-trace budget T it is budget-forced.
3. **Exchange** into every still-active instance: `[PEER] ... [/PEER]` followed by a blank line.
   Each instance's own paragraphs carry a forced `[ME]` tag and the peer markers are stop
   sequences, because without this the model started mimicking `[PEER]` on its own paragraphs
   and fabricating peer text (seen in the smoke test).
   `raw` pastes the peer's chunk. `summary`, `answer` and `critique` are produced by an extra
   call in the peer's context (temperature 0.3, ≤120 tokens): a cue such as "[Summary of my
   reasoning so far for my peer, at most 3 sentences ...]" appended to the peer's own text; for
   `critique` the cue is appended to *my* text and the peer answers it. Extra tokens are counted.
   In `s_` mode there are no markers or tags: the insert is a plain paragraph in the first person.
   `s_summary` asks the peer for a recap with a LeaP-style first-person trigger and pastes it after "Wait,
   let me consider a different line of reasoning:". `s_challenge` (and `x_critique`) are written by an
   extra call on *my* text plus a hidden note holding the peer's own text; `s_challenge` must start with
   "Wait," and must not mention the other attempt (inserts that do are counted as leaks). The note is never
   inserted, only the challenge.
   A peer that has finished sends one last insert and then goes quiet.
4. Repeat until both are done. Each instance then produces its answer (think mode: close the
   block and answer in ≤1000 tokens; both modes: if no `\boxed{}`, force "Therefore, the final
   answer is \boxed{" and 60 greedy tokens). Identical readout in every condition.

The system prompt tells each instance it is one of two instances solving the same problem in its
own scratchpad, what the `[PEER]` material is, that it may be right or wrong, and to continue its
own reasoning. `x_none` gets the same prompt without the peer sentence.

Secondary design kept for comparison: `twin_asym` / `twin_sym`, two instances alternating
paragraph by paragraph on ONE shared trace (solver/checker roles or identical roles).

Scoring: last `\boxed{}` by brace matching, LaTeX normalisation with numeric fallback.
`src/summarize.py` reports everything in section 1 with paired bootstrap CIs over problems.

### 4.1 Infrastructure notes

- Generation ~70 tok/s per stream, prompt processing ~1700 tok/s uncached; prefix cache hits are
  near instant and survive alternating prompts across the two slots.
- The MLX engine process **segfaulted** twice today under load ("The model has crashed without
  additional information"), after which every request gets HTTP 400 "No models loaded". The
  harness has a watchdog that reloads the model and retries the request.
- A `\boxed{}` inside a pasted insert is never read as the receiver's answer: in no-think mode the
  forced-box readout checks only the instance's own paragraphs.
- Every generation point ends on a word, never on a space ("Wait," not "Wait, "; "[ME]" not "[ME] ").
  Qwen word tokens carry their own leading space, so a prompt ending in a space pushed the model towards
  space-less tokens: in the first smoke tests nearly every insert started with a digit ("Wait, 1 is ...",
  "[PEER] 1. The total ..."), and every `[ME]` paragraph was generated from that same skewed state.
- No-think forced readout: "I must stop here and commit to the best answer I have." then the box, one line
  only. Without the line stop the model escaped the box with a blank line and kept reasoning.
- An instance within 16 tokens of its budget is out of budget, and one that produces no tokens is stopped
  ("stall"). Before this, an instance one token short of its budget looped on empty chunks until the
  60-round cap; both server wedges today followed such bursts of zero-token requests.
- On a request timeout the watchdog unloads and reloads the model (the engine wedges with requests stuck
  in "processing prompt"; it has also crashed outright, which the same watchdog handles).
- Request timeout is 300 s with logged retries. With the earlier 1800 s timeout a smoke job sat hung for
  2.7 hours.
- Readout had to be fixed early: the model restates the whole solution in the visible answer and
  ran past a 600-token cap before boxing. Now 1000 tokens plus the forced-box fallback.
- Grader (2026-10-26): `is_equiv` marked four MATH answer formats wrong: units in `\text{}` (5.4 vs
  "5.4 \text{ cents}"), `\$32,\!348`, a tuple without parentheses, and an inequality for an interval
  (`3 < \lambda \le 4` vs `(3,4]`). Fixed, and src/regrade.py re-grades stored predictions (29/280 main
  records and 12/240 screen records changed). The selection had used the old grader, so four "contested"
  problems were really 3/3; strata were recomputed from the regraded screen (17 contested, 14 easy).

### 4.2 Calibration (4 level-5 problems, seed 0, shared-trace design)

Think mode, T=4000: single 4/4 but budget-forced in 3/4; shared-trace twin 4/4, closed its own
trace at ~2.9k tokens. No-think, T=2500: single 3/4, twin 4/4, ~1.4k tokens each, twin 38 s vs 23 s.
The miss was a forgotten root exclusion (9π vs 8π) that the checker turn caught. No-think is 3x
faster and has headroom, so it is the primary track.

### 4.3 Main run design (src/run_main.sh)

Throughput: 4 server slots give 1.46x the aggregate decode rate of 2 (202 vs 138 tok/s; 51 vs 70 tok/s
per stream), so two two-trace jobs run at once. A two-trace job takes ~2-4 min, 4-6x its pure decode
time; the overhead is the per-round synchronisation and the extra insert calls.

1. Screen: `single` x seeds 0-2 on 80 level-5 problems (MATH-500, shuffle seed 0), T=2500, no-think.
2. Select (src/select_problems.py): every problem not solved 3/3 ("contested") plus 10 random 3/3 ones
   ("easy", so that herding damage on easy problems stays visible).
3. `single` on FRESH seeds 3-5 on the selected set; the screening seeds were used for selection and would
   regress to the mean.
4. Two-trace conditions x_none, x_summary, x_critique, s_wait, s_summary, s_challenge, seed 0 (more seeds
   later via `TWO_SEEDS=1 src/run_main.sh`, which resumes).

## 5. Predictions (written before the run)

- P1: `x_none` per-instance ≈ `single` (identity argument). A gap means the "two instances"
  framing or the chunked generation itself changes behaviour.
- P2: `x_raw` per-instance > `single` by a few points if error correction (a) dominates; ≤ if
  distraction (d) dominates. Hogwild's small-model result says ≤ is the honest prior.
- P3: `x_summary` ≥ `x_raw`: shorter inserts, less distraction, same signal. `x_answer` carries the
  least information and should herd most (agreement up, pair accuracy not).
- P4: pair accuracy of any x_ condition vs `single-pair`: the agreement rate will rise under
  exchange (herding) but pair accuracy will move little. Beating `single-pair` by a clear margin
  at equal tokens would be the interesting result.
- P5: `sc@3` ≥ every pair readout, at 1.5x the pair's cost.
- P6 (framing): `s_summary` ≥ `x_summary` and `s_challenge` ≥ `x_critique` per instance. First-person
  material uses the model's trained self-correction path, avoids the persona tax and cannot be dismissed
  as "someone else's". The risk is the mirror image: a confident-sounding wrong insert is harder to
  discount when it reads as one's own conclusion, so herding (agreement) should be highest in `s_`.
- P7: `s_wait` ≈ `single` in no-think mode; a fixed challenge with no new information mostly adds
  re-checking tokens. If `s_challenge` beats `s_wait`, the peer's information is doing the work.

## 6. Results: main run, two-trace seed 0 (2026-10-26)

Setting: 31 MATH-500 level-5 problems (17 contested, 14 easy after regrading, see 4.1), no-think,
T=2500 own tokens per trace, chunks of 150-350 tokens, an exchange after every chunk (~10 per trace).
`single` on 3 fresh seeds, each two-trace condition on 1 seed. Deltas are paired over problems with
bootstrap 95% CIs. Data: results/main_x (`src/summarize.py results/main_x`).

| cond | per-instance | vs single | pair | vs single-pair | agree | pass@2 | budget hit | tokens/job |
|---|---|---|---|---|---|---|---|---|
| single (pair = 2 indep. seeds) | 0.645 | | 0.667 | | 0.68 | 0.72 | 0.45 | 2033 (x2) |
| x_none | 0.629 | -0.02 [-0.16, +0.12] | 0.677 | +0.01 [-0.13, +0.15] | 0.68 | 0.68 | 0.58 | 4147 |
| x_summary | 0.677 | +0.03 [-0.05, +0.13] | 0.645 | -0.02 [-0.13, +0.10] | 0.84 | 0.71 | 0.53 | 5869 |
| x_critique | 0.677 | +0.03 [-0.07, +0.13] | 0.677 | +0.01 [-0.09, +0.11] | 0.81 | 0.68 | 0.45 | 5298 |
| s_wait | 0.129 | **-0.52** [-0.68, -0.34] | 0.161 | **-0.51** [-0.67, -0.33] | 0.39 | 0.19 | 0.97 | 4865 |
| s_summary | 0.500 | **-0.15** [-0.27, -0.02] | 0.452 | **-0.22** [-0.36, -0.09] | 0.74 | 0.55 | 0.44 | 4222 |
| s_challenge | 0.548 | -0.10 [-0.24, +0.05] | 0.452 | **-0.22** [-0.37, -0.07] | 0.48 | 0.71 | 0.71 | 6470 |

`sc@3` of single (3 seeds, ~6100 tokens) is 0.710, above every pair readout.

Framing contrasts (same information, first person vs [PEER] block): s_summary - x_summary -0.18
[-0.32, -0.05] per instance; s_challenge - x_critique -0.13 [-0.29, +0.05] per instance and -0.23
[-0.39, -0.07] pair. Information contrast: s_challenge - s_wait +0.42 [+0.26, +0.58], s_summary - s_wait
+0.37 [+0.19, +0.55].

What happens to the pair (fractions of pairs; single-pair = two independent seeds):

| cond | agree, right | agree, wrong | split, one right | split, both wrong |
|---|---|---|---|---|
| single-pair | 0.57 | 0.11 | 0.15 | 0.17 |
| x_none | 0.58 | 0.10 | 0.10 | 0.23 |
| x_summary | 0.65 | 0.19 | 0.06 | 0.10 |
| x_critique | 0.68 | 0.13 | 0.00 | 0.19 |
| s_wait | 0.06 | 0.32 | 0.13 | 0.48 |
| s_summary | 0.45 | 0.29 | 0.10 | 0.16 |
| s_challenge | 0.35 | 0.13 | 0.32 | 0.16 |

### 6.1 Predictions

- P1 holds: `x_none` ≈ `single`. Chunked generation and the two-instance prompt cost nothing measurable,
  though they raise the budget-hit rate (0.58 vs 0.45).
- P2, P3: not tested (x_raw, x_answer not run).
- P4 holds: exchange raises agreement (0.68 -> 0.84 / 0.81) and leaves pair accuracy where it was. It
  resolves the one-right splits (0.15 -> 0.06 / 0.00), but toward the wrong answer about as often as
  the right one; on the contested stratum x_summary agrees on a wrong answer in 0.35 of pairs vs 0.20.
- P5 holds: `sc@3` (0.710) beats every pair readout at similar token cost.
- P6 rejected. Self framing is worse than peer framing with the same information, and the risk named in
  P6 is what happened: a peer's claim inserted as one's own thought is accepted, not checked (s_summary
  agrees on a wrong answer in 0.29 of pairs vs 0.11).
- P7 rejected in its first half: a fixed "Wait, let me challenge ..." after every chunk is catastrophic,
  not neutral. The instance re-opens its work every ~200 tokens and never finishes (97% hit the budget),
  and its forced answers are poor; it fails even on the easy stratum. Second half: s_challenge and
  s_summary beat s_wait by ~0.4, so the peer's material is informative relative to an empty challenge,
  but neither beats inserting nothing.

### 6.2 Mechanisms seen in the traces

1. **Herding through first-person summaries.** intermediate_algebra/1462 (single 3/3): A writes
   `\boxed{661}` in its own text; B receives "Wait, let me consider a different line of reasoning: the sum
   is $\boxed{661}$." and finishes with 661 at once. counting_and_probability/181 (single 3/3): A's
   "7n = 70, so n = 10" reaches B the same way and B adopts it.
2. **Challenges that decorrelate without correcting.** s_challenge has the most one-right splits (0.32)
   and the highest pass@2 together with x_summary, so it does push the instances apart. But the generated
   challenge often misses the peer's decisive fact: in counting_and_probability/188 B had the right 1/2,
   yet A's challenges re-verified a correct sub-area and never questioned the wrong one; in 181 the
   challenge re-opened the grid interpretation four times. The pair rule picks the wrong side in 8 of 10
   one-right splits. In 5 of them one instance finished on its own and it was the wrong one 4 times: the
   instance still being challenged keeps re-checking, and the one that stops is the one that settled. In
   the other 5 both hit the budget and the fallback (A) was wrong 4 times, which is chance.
3. **Format break, minor.** In 2 of 62 s_challenge traces the model writes `</think>` mid-answer and
   restarts the solution, as if the run of "Wait," paragraphs had been a thinking block.

Caveats: one seed per two-trace condition, n = 31, insertion after every chunk only. The x_ deltas are
nulls with CIs of about +-0.1, not evidence of no effect.

### 6.3 Follow-up: sparse exchange and a second seed (src/run_sparse.sh, 2026-10-26 00:11-07:46)

`cond@4` exchanges after every 4th round, about every 700 tokens, so about 1.7 inserts per trace instead
of about 8. That is closer to LeaP's cadence. Dense conditions now have seeds 0 and 1 (s_wait seed 0
only); sparse ones have seed 0. Deltas are against `single` (per instance) and `single-pair` (pair).

| cond | per-instance | vs single | pair | vs single-pair | agree, wrong | contested per-instance vs single |
|---|---|---|---|---|---|---|
| x_summary (2 seeds) | 0.718 | +0.07 [-0.01, +0.17] | 0.710 | +0.04 [-0.05, +0.14] | 0.15 | +0.14 [-0.02, +0.31] |
| x_summary@4 | 0.694 | +0.05 [-0.09, +0.18] | 0.710 | +0.04 [-0.10, +0.17] | 0.23 | +0.07 [-0.16, +0.29] |
| x_critique (2 seeds) | 0.661 | +0.02 [-0.06, +0.09] | 0.661 | -0.01 [-0.09, +0.08] | 0.16 | |
| x_critique@4 | 0.629 | -0.02 [-0.11, +0.08] | 0.677 | +0.01 [-0.09, +0.11] | 0.06 | +0.01 [-0.12, +0.16] |
| s_summary (2 seeds) | 0.476 | **-0.17** [-0.29, -0.05] | 0.452 | **-0.22** [-0.34, -0.10] | 0.26 | |
| s_summary@4 | 0.597 | -0.05 [-0.19, +0.09] | 0.613 | -0.05 [-0.22, +0.12] | 0.13 | +0.10 [-0.05, +0.27] |
| s_challenge (2 seeds) | 0.589 | -0.06 [-0.17, +0.05] | 0.548 | **-0.12** [-0.23, -0.02] | 0.11 | |
| s_challenge@4 | 0.694 | +0.05 [-0.08, +0.18] | **0.742** | +0.08 [-0.03, +0.19] | 0.10 | +0.16 [-0.05, +0.37] |
| s_wait | 0.129 | **-0.52** | 0.161 | **-0.51** | 0.32 | |
| s_wait@4 | 0.484 | **-0.16** [-0.31, -0.02] | 0.516 | **-0.15** [-0.29, -0.01] | 0.16 | -0.17 [-0.40, +0.08] |

single-pair: agree & wrong 0.11; sc@3 of single 0.710.

- **Cadence matters for the self framing and not for the peer framing.** Sparse minus dense, per
  instance: s_wait +0.36 [+0.18, +0.53], s_summary +0.12 [-0.02, +0.27], s_challenge +0.11 [+0.00,
  +0.21] (pair +0.19 [+0.05, +0.34]); x_summary -0.02, x_critique -0.03 (both ≈ 0). A [PEER] block can
  be discounted however often it arrives. A first-person insert cannot, so each extra one is either
  adopted (herding) or reopens the work (budget).
- **s_challenge@4 looked like the best two-copy condition on seed 0; it did not replicate (6.4).** Its pair readout is 0.742, above
  single-pair (0.667) and sc@3 (0.710), at 4870 tokens against ~6100 for sc@3. It is the only condition
  that raises "agree, right" (0.65 vs 0.57) without raising "agree, wrong" (0.10 vs 0.11). But it is one
  seed and every CI includes 0. Dense s_challenge moved from 0.452 to 0.645 between seeds 0 and 1, so a
  single-seed condition can move by +-0.1 on its own.
- With the second seed, the dense framing contrast got stronger: s_summary - x_summary -0.24
  [-0.38, -0.10] per instance. x_summary - x_none is +0.10 [+0.00, +0.19], the first contrast that
  touches significance for peer-framed exchange.
- A bare "Wait" challenge stays harmful even when sparse (s_wait@4 -0.16). With the same cadence, the
  peer-informed challenge is +0.21 [+0.10, +0.34] better (s_challenge@4 - s_wait@4), so the content
  of the challenge carries the effect, not the act of challenging.

### 6.4 Replication: the s_challenge@4 lead does not hold (src/run_rep.sh, 2026-10-26 08:19-12:32)

| cond | pair acc per seed | pair vs single-pair | per-instance vs single |
|---|---|---|---|
| s_challenge@4 (3 seeds) | 0.742 0.452 0.548 | **-0.09** [-0.16, -0.01] | -0.06 [-0.15, +0.03] |
| x_critique@4 (3 seeds) | 0.677 0.677 0.613 | -0.01 [-0.09, +0.08] | +0.00 [-0.07, +0.08] |
| s_challenge@8 (1 seed, ~0.7 inserts per trace) | 0.710 | +0.04 [-0.04, +0.13] | +0.03 [-0.05, +0.12] |
| single (3 seeds; pair = 2 indep. seeds) | 0.677 0.613 0.645 per instance | | |

- Seed 0 of s_challenge@4 was a lucky draw. Seeds 1 and 2 used the same tokens, inserts and leak counts
  (4600-4870 tokens/job, 3.4-3.6 inserts/job), so this is not an infrastructure artifact.
- The self-challenge raises outcome variance. 12 of 31 problems have a mixed pair outcome over three
  seeds, against 9 for x_critique@4 and 7 for single. The seed-to-seed sd of pair accuracy is 0.12, against
  0.03 for single and x_critique@4. One-seed results for s_ conditions are therefore unreliable.
- s_challenge@8 is close to no interlacing at all (43 inserts over 31 jobs), so its +0.04 says little.
- Overall: in this setting no exchange algorithm beats independent sampling. Peer-framed exchange is
  neutral (x_summary is the best, +0.07 [-0.01, +0.17] per instance over 2 seeds, consistent with LeaP's
  direction but not significant here). Self-framed exchange is harmful when dense and, at best, neutral but
  noisier when sparse. sc@3 of single (0.710) remains the best readout per token.

## 7. Follow-ups

1. Done (6.4): s_challenge@4 did not replicate. Open: one-directional flow (only A receives); more
   seeds of x_summary, the only peer-framed exchange with a consistently positive (not significant) sign.
2. Asymmetric x_: A explores at temperature 1.0, B at 0.3; or solver/checker roles.
3. A verification-aware pair readout: a third call that sees both traces and must verify. The pair
   rule "prefer the instance that finished on its own" fails under s_challenge (6.2).
4. Think-mode track (native thinking) at T=4000 on the same problems.
5. Harder set (AIME 2025, many seeds) once the MATH-500 L5 signal is understood.
6. Exchange only on disagreement: the instances post candidate answers, and material is inserted only
   when they differ, so agreeing pairs are left alone and herding can only act on real splits.
