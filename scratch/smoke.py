import time, json, httpx
B="http://localhost:1234/v1"
def comp(prompt, max_tokens=200, stop=None, temperature=0.7):
    t=time.time()
    r=httpx.post(B+"/completions", json={"model":"twin","prompt":prompt,"max_tokens":max_tokens,"temperature":temperature,"stop":stop or []}, timeout=600).json()
    dt=time.time()-t
    ch=r["choices"][0]; u=r.get("usage",{})
    return ch["text"], ch.get("finish_reason"), u, dt
sys_a="You are a careful mathematician."
q="What is the sum of the first 10 positive odd integers?"
# 1) raw ChatML with open <think>
p=f"<|im_start|>system\n{sys_a}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
txt,fr,u,dt=comp(p,300,stop=["\n\n"])
print("== raw think, stop on blank line ==");print(repr(txt[:400]));print(fr,u,f"{dt:.1f}s", f"{u.get('completion_tokens',0)/dt:.0f} tok/s")
# 2) raw ChatML with closed empty think (no-think mode)
p2=f"<|im_start|>system\n{sys_a}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
txt,fr,u,dt=comp(p2,300)
print("== raw no-think ==");print(repr(txt[:400]));print(fr,u,f"{dt:.1f}s", f"{u.get('completion_tokens',0)/dt:.0f} tok/s")
# 3) chat endpoint default, check reasoning fields
t=time.time()
r=httpx.post(B+"/chat/completions", json={"model":"twin","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0.7}, timeout=600).json()
m=r["choices"][0]["message"]; print("== chat default =="); print({k:(v[:150] if isinstance(v,str) else v) for k,v in m.items()}); print(r.get("usage"), f"{time.time()-t:.1f}s")
# 4) chat endpoint with enable_thinking false
t=time.time()
r=httpx.post(B+"/chat/completions", json={"model":"twin","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}}, timeout=600).json()
m=r["choices"][0]["message"]; print("== chat enable_thinking=False =="); print({k:(v[:150] if isinstance(v,str) else v) for k,v in m.items()}); print(r.get("usage"), f"{time.time()-t:.1f}s")
