"""Registry of every experiment of the NeedNbr paper -> scheduler jobs."""
import os

from .data import PROC
from .scheduler import Job, py
from .utils import ROOT, load_config, slug

R = ROOT / "results"


def gated_accessible(model):
    """True if a gated HF model can actually be downloaded with the current HF_TOKEN (checked once per model)."""
    import urllib.request
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return False
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    req = urllib.request.Request(f"{base}/{model}/resolve/main/config.json", method="HEAD",
                                 headers={"Authorization": f"Bearer {tok}"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return True
    except Exception:
        return False


def build_jobs(config_path=None, quick=False):
    """Job order = priority (the scheduler starts the first runnable job): data prep + GNN baselines, then for each
    LLM its main job per dataset (the largest graph last), then the k-sensitivity runs, the 3B scaling runs, the 32B
    scaling runs last (they need both GPUs at once). CPU analysis jobs run as soon as their LLM job is done."""
    cfg = load_config(config_path)
    c = ["--config", config_path] if config_path else []
    M = cfg["models"]
    llms = list(M["llms"])
    gated = [m for m in llms if m.startswith("meta-llama/") and not gated_accessible(m)]
    if gated:
        print(f"[experiments] no access to gated models {gated} (HF_TOKEN missing or access not granted) -> skipped", flush=True)
    llms = [m for m in llms if m not in gated]
    datasets = list(cfg["main_datasets"])
    if quick:
        llms, datasets = llms[:1], datasets[:3]
    big = [d for d in ("arxiv",) if d in datasets]
    order = [d for d in datasets if d not in big] + big
    J = []
    for ds in datasets:
        J.append(Job(f"prep_{ds}", py("needn.prepare", "--dataset", ds, *c), PROC / f"{ds}_emb.npy"))
        J.append(Job(f"gnn_{ds}", py("needn.gnn", "--dataset", ds, *c), R / "gnn" / f"{ds}.json", [f"prep_{ds}"]))
        # GCN class probabilities for the LLM+GNN ensemble / cascade baselines (minutes; read by analyze if present)
        J.append(Job(f"gnnprob_{ds}", py("needn.gnn", "--dataset", ds, "--probs", *c), R / "gnn" / f"probs_{ds}.npy",
                     [f"prep_{ds}"]))
    for i, m in enumerate(llms):
        for ds in order:
            ann = ["--annotate", 2000] if (i == 0 and ds in cfg.get("distill_datasets", []) and not quick) else []
            icl = ["--graphicl"] if i < cfg.get("graphicl_llms", 1) else []  # GraphICL-style baseline (first LLM)
            name = f"llm_{slug(m)}_{ds}"
            J.append(Job(name, py("needn.run_llm", "--dataset", ds, "--llm", m, *ann, *icl, *c),
                         R / "llm" / slug(m) / ds / "raw.pkl", [f"prep_{ds}"]))
            J.append(Job(f"analyze_{slug(m)}_{ds}", py("needn.analyze", "--dataset", ds, "--llm", m, *c),
                         R / "analysis" / slug(m) / f"{ds}.json", [name], n_gpus=0, always=True))
            if ann:
                J.append(Job(f"distill_{ds}", py("needn.distill", "--dataset", ds, "--llm", m, *c),
                             R / "distill" / slug(m) / f"{ds}.json", [name]))
    # k-sensitivity (first LLM): re-uses f0 / SIG of the main run -> depends on it. The k table is built by
    # aggregate.py from whatever raw_k*.pkl exist, so a failed k-run never blocks any other result.
    if llms and not quick:
        m = llms[0]
        for ds in [d for d in cfg.get("k_sens_datasets", []) if d in datasets]:
            name = f"llm_{slug(m)}_{ds}"
            for kk in cfg.get("k_sens_values", []):
                J.append(Job(f"{name}_k{kk}", py("needn.run_llm", "--dataset", ds, "--llm", m, "--k", kk, *c),
                             R / "llm" / slug(m) / ds / f"raw_k{kk}.pkl", [name]))
    # scaling runs (same model family 3B / 7B / 32B): the small model on one GPU, the 32B-AWQ last on both GPUs
    strat = ["--strategies", ",".join(cfg.get("scale_strategies", ["random", "similar", "cns"]))]
    for key, ng in (("small_llm", 1), ("large_llm", 2)):
        m = M.get(key)
        if quick or not m:
            continue
        for ds in [d for d in cfg.get("scale_datasets", []) if d in datasets]:
            name = f"llm_{slug(m)}_{ds}"
            J.append(Job(name, py("needn.run_llm", "--dataset", ds, "--llm", m, *strat, *c),
                         R / "llm" / slug(m) / ds / "raw.pkl", [f"prep_{ds}"], n_gpus=ng))
            J.append(Job(f"analyze_{slug(m)}_{ds}", py("needn.analyze", "--dataset", ds, "--llm", m, *c),
                         R / "analysis" / slug(m) / f"{ds}.json", [name], n_gpus=0, always=True))
    return J
