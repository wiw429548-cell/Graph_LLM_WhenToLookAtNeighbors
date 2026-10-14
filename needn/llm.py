"""Batched chat LLM with a persistent on-disk cache (resumable runs).

backend 'vllm' (default on the server) or 'hf' (transformers; used for local smoke tests / when vLLM is absent).
Every (model, prompt, parameters) is cached in data/cache/<model>/<cache_name>.sqlite (one file per dataset, so
parallel jobs never write the same SQLite file -- important on NFS). Results are committed every `CHUNK` prompts,
so a crashed job resumes where it stopped.

Prompts are rendered with the model's chat template and passed to the engine as token ids (no second BOS token).
option_probs() reads the distribution over the answer letters from the first generated token:
  P(option l) = sum over the vocabulary tokens that spell l (e.g. 'A' and ' A'), renormalised over the options.
"""
import hashlib
import json
import math
import os
import sqlite3
import threading

from .runtime import build_vllm, setup_env
from .utils import DATA, ROOT, slug

setup_env()  # must run before vLLM is imported anywhere in this process

CHUNK = int(os.environ.get("NEEDN_CACHE_CHUNK", "2048"))
NEG = -1e9


class Cache:
    def __init__(self, model, name="default"):
        p = DATA / "cache" / slug(model) / f"{slug(name)}.sqlite"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self.db = sqlite3.connect(str(p), check_same_thread=False, timeout=600)
        self.db.execute("PRAGMA journal_mode=DELETE")  # WAL needs shared memory -> unsafe on NFS
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()
        self.lock = threading.Lock()

    def get_many(self, keys):
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT k, v FROM c WHERE k IN (%s)" % ",".join("?" * len(chunk))
            for k, v in self.db.execute(q, chunk):
                out[k] = v
        return out

    def put_many(self, items):
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO c VALUES (?, ?)", items)
            self.db.commit()


class LLM:
    def __init__(self, model, backend=None, max_model_len=8192, gpu_mem=0.85, tp=1, dtype="auto",
                 enable_thinking=False, seed=0, cache_name="default", max_logprobs=80, log=print):
        self.model = model
        self.backend = backend or os.environ.get("LLM_BACKEND") or self._default_backend()
        self.cache = Cache(model, cache_name)
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.max_logprobs = max_logprobs
        self.log = log
        self.stats = {"option_calls": 0, "option_no_letter_in_topk": 0, "truncated_prompts": 0}
        self.engine_info = {"backend": self.backend, "model": model, "max_model_len": max_model_len}
        if self.backend == "vllm":
            # TP clamp + free-memory-aware gpu_mem + retry ladder; sets VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc needed)
            self.engine, self.engine_info = build_vllm(
                model, tp=tp, max_model_len=max_model_len, gpu_mem=gpu_mem, dtype=dtype, trust_remote_code=True,
                seed=seed, enable_prefix_caching=True, max_logprobs=max_logprobs,
                log_file=ROOT / "results" / "engine_log.jsonl")
            self.engine_info["backend"] = "vllm"
            max_model_len = self.engine_info["max_model_len"]
            try:
                self.tok = self.engine.get_tokenizer()
            except Exception:  # noqa: BLE001  (API differences between vLLM versions)
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"
            dt = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available()
                  else torch.float32)
            self.hf = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=dt,
                trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.max_model_len = max_model_len
        self._letter_cache = {}

    @staticmethod
    def _default_backend():
        import importlib.util
        return "vllm" if importlib.util.find_spec("vllm") else "hf"

    # ------------------------------------------------------------------ prompt rendering
    def render(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if getattr(self.tok, "chat_template", None):
            kw = {"enable_thinking": self.enable_thinking} if "qwen3" in self.model.lower() else {}
            try:
                return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            except Exception:  # noqa: BLE001  e.g. templates without a system role (Gemma, old Mistral)
                if messages and messages[0]["role"] == "system":
                    merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                    return self.tok.apply_chat_template(merged + messages[2:], tokenize=False,
                                                        add_generation_prompt=True, **kw)
                raise
        return "\n\n".join(m["content"] for m in messages) + "\n\n"

    def encode(self, rendered):
        """token ids of a rendered prompt. The chat template already contains BOS (if the model uses one), so no
        special tokens are added here (vLLM would otherwise add a second BOS for e.g. Llama-3)."""
        return self.tok(rendered, add_special_tokens=False)["input_ids"]

    def _fit(self, ids, max_new):
        budget = self.max_model_len - max_new - 8
        if len(ids) <= budget:
            return ids
        self.stats["truncated_prompts"] += 1  # keep the head (instructions) and the tail (question)
        head = budget // 3
        return list(ids[:head]) + list(ids[-(budget - head):])

    # ------------------------------------------------------------------ generation
    def generate(self, prompts, temperature=0.0, max_tokens=256, n=1, stop=None, tag=""):
        """prompts: list of chat messages (or str). returns list (len(prompts)) of list (n) of strings."""
        rendered = [self.render(p) for p in prompts]
        params = json.dumps([temperature, max_tokens, stop, self.seed, tag])
        keys = [[hashlib.sha1((r + params + f"#{j}").encode()).hexdigest() for j in range(n)] for r in rendered]
        flat = [k for ks in keys for k in ks]
        hit = self.cache.get_many(flat)
        todo = [i for i, ks in enumerate(keys) if any(k not in hit for k in ks)]
        for s in range(0, len(todo), CHUNK):
            part = todo[s:s + CHUNK]
            outs = self._run([self._fit(self.encode(rendered[i]), max_tokens) for i in part], temperature, max_tokens,
                             n, stop)
            items = []
            for i, o in zip(part, outs):
                for j, txt in enumerate(o):
                    hit[keys[i][j]] = txt
                    items.append((keys[i][j], txt))
            self.cache.put_many(items)
        return [[hit[k] for k in ks] for ks in keys]

    # ------------------------------------------------------------------ option scoring (classification)
    def option_probs(self, prompts, letters, tag=""):
        """distribution over answer options from the first generated token.
        prompts: chat messages; letters: list of option symbols (e.g. ['A','B',...]) -> np.ndarray [N, len(letters)]"""
        import numpy as np
        rendered = [self.render(p) for p in prompts]
        params = json.dumps(["optprob", letters, tag])
        keys = [hashlib.sha1((r + params).encode()).hexdigest() for r in rendered]
        hit = self.cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in hit]
        for s in range(0, len(todo), CHUNK):
            part = todo[s:s + CHUNK]
            res = self._option_logprobs([self._fit(self.encode(rendered[i]), 1) for i in part], letters)
            n_miss = sum(all(x <= NEG / 2 for x in v) for v in res)
            self.stats["option_calls"] += len(res)
            self.stats["option_no_letter_in_topk"] += n_miss
            if len(res) >= 50 and n_miss > 0.5 * len(res):
                raise RuntimeError(f"[llm] {tag}: for {n_miss}/{len(res)} prompts no option letter was among the "
                                   f"top-{self.max_logprobs} first tokens -> the option distribution would be "
                                   f"meaningless; check the prompt / chat template (e.g. thinking mode).")
            items = []
            for i, v in zip(part, res):
                hit[keys[i]] = json.dumps(v)
                items.append((keys[i], hit[keys[i]]))
            self.cache.put_many(items)
        out = np.array([json.loads(hit[k]) for k in keys], dtype=np.float64).reshape(len(keys), len(letters))
        out = np.exp(out - out.max(1, keepdims=True))
        out[np.isnan(out)] = 0
        s = out.sum(1, keepdims=True)
        return np.where(s > 0, out / np.maximum(s, 1e-12), 1.0 / len(letters))

    def letter_token_ids(self, letters):
        """{token id: option index} for the single-token spellings 'X' and ' X' of every option letter."""
        key = tuple(letters)
        if key not in self._letter_cache:
            m = {}
            for i, l in enumerate(letters):
                for variant in (l, " " + l):
                    ids = self.tok(variant, add_special_tokens=False)["input_ids"]
                    if len(ids) == 1 and ids[0] not in m:
                        m[ids[0]] = i
            self._letter_cache[key] = m
        return self._letter_cache[key]

    def _decode_letter(self, tid, obj, idx):
        t = getattr(obj, "decoded_token", None)
        if t is None:  # some vLLM versions do not detokenize logprob tokens
            t = self.tok.decode([tid])
        t = t.replace("\u2581", " ").strip().rstrip(".):")
        return idx.get(t)

    def _option_logprobs(self, prompt_ids, letters):
        """returns list of [len(letters)] log-probs (NEG if no spelling of the option is among the top tokens)"""
        idx = {l: i for i, l in enumerate(letters)}
        tid2opt = self.letter_token_ids(letters)
        res = []
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=self.max_logprobs)
            outs = self.engine.generate([{"prompt_token_ids": list(p)} for p in prompt_ids], sp,
                                        use_tqdm=len(prompt_ids) > 200)
            for r in outs:
                acc = [[] for _ in letters]
                lps = r.outputs[0].logprobs
                lp = lps[0] if lps else {}
                for tid, obj in (lp.items() if hasattr(lp, "items") else []):
                    j = tid2opt.get(int(tid))
                    if j is None:
                        j = self._decode_letter(int(tid), obj, idx)
                    if j is not None:
                        acc[j].append(float(obj.logprob))
                res.append([_logsumexp(a) for a in acc])
            return res
        import torch
        groups = [[] for _ in letters]
        for tid, j in tid2opt.items():
            groups[j].append(tid)
        for s in range(0, len(prompt_ids), 8):
            batch = prompt_ids[s:s + 8]
            L = max(len(p) for p in batch)
            pad = self.tok.pad_token_id
            ids = torch.tensor([[pad] * (L - len(p)) + list(p) for p in batch], device=self.hf.device)
            mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in batch], device=self.hf.device)
            pos = (mask.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self.hf(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1]
            logp = logits.float().log_softmax(-1).cpu()
            for row in logp:
                res.append([_logsumexp([row[t].item() for t in g]) if g else NEG for g in groups])
        return res

    def n_tokens(self, prompts):
        return [len(self.encode(self.render(p))) for p in prompts]

    def chat(self, prompts, **kw):
        return [o[0] for o in self.generate(prompts, **kw)]

    def _run(self, prompt_ids, temperature, max_tokens, n, stop):
        if self.backend == "vllm":
            from vllm import SamplingParams
            n_req = 1 if temperature == 0 else n  # some vLLM versions reject n > 1 with greedy decoding
            sp = SamplingParams(temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                                max_tokens=max_tokens, n=n_req, stop=stop, seed=self.seed)
            res = self.engine.generate([{"prompt_token_ids": list(p)} for p in prompt_ids], sp,
                                       use_tqdm=len(prompt_ids) > 200)
            return [[c.text for c in r.outputs] * (n // n_req) for r in res]
        import torch
        out = []
        bs = 8
        pad = self.tok.pad_token_id
        for s in range(0, len(prompt_ids), bs):
            batch = prompt_ids[s:s + bs]
            L = max(len(p) for p in batch)
            ids = torch.tensor([[pad] * (L - len(p)) + list(p) for p in batch], device=self.hf.device)
            mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in batch], device=self.hf.device)
            with torch.no_grad():
                g = self.hf.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens,
                                     do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                     num_return_sequences=n, pad_token_id=pad)
            txt = self.tok.batch_decode(g[:, L:], skip_special_tokens=True)
            for b in range(len(batch)):
                outs = txt[b * n:(b + 1) * n]
                if stop:
                    outs = [min((o.split(st)[0] for st in stop), key=len) for o in outs]
                out.append(outs)
        return out


def _logsumexp(xs):
    xs = [x for x in xs if x > NEG / 2]
    if not xs:
        return NEG
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
