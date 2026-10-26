"""Aggregate decode throughput with N concurrent streams (run after reloading with --parallel N)."""
import sys, time, httpx, concurrent.futures as cf
N = int(sys.argv[1]); B = "http://localhost:1234/v1"
p = ("<|im_start|>user\nWrite a long, detailed essay about the history of mathematics, at least 2000 words."
     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
def one(i):
    t = time.time()
    r = httpx.post(B + "/completions", json={"model": "twin", "prompt": p + f"Part {i}:", "max_tokens": 400,
                                             "temperature": 0.7, "seed": i}, timeout=600).json()
    return r["usage"]["completion_tokens"], time.time() - t
t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex: res = list(ex.map(one, range(N)))
dt = time.time() - t0; full = [(c, t) for c, t in res if c >= 390]
print(f"N={N}: streams (tokens, s) {[(c, round(t, 1)) for c, t in res]}; "
      f"full-length streams: {len(full)}, per-stream {sum(c / t for c, t in full) / max(1, len(full)):.0f} tok/s, "
      f"aggregate {sum(c for c, _ in res) / dt:.0f} tok/s")
