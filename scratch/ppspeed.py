import time, httpx
B="http://localhost:1234/v1"
filler=("Consider the polynomial p(x)=x^4-3x^2+2. Its roots are ±1 and ±√2. We verify: p(1)=1-3+2=0. p(√2)=4-6+2=0. ")*120
for sysmsg in ["SYS-A: You are solver.", "SYS-B: You are checker.", "SYS-A: You are solver."]:
    p=f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{filler}\nWhat is p(0)?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    t=time.time()
    r=httpx.post(B+"/completions", json={"model":"twin","prompt":p,"max_tokens":8,"temperature":0}, timeout=600).json()
    dt=time.time()-t; u=r["usage"]
    print(sysmsg[:5], "prompt_tokens",u["prompt_tokens"], f"{dt:.2f}s", f"{u['prompt_tokens']/dt:.0f} tok/s (incl. 8 gen)")
