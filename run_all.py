"""One-command entry point for ALL experiments of the NeedNbr paper.

    CUDA_VISIBLE_DEVICES=0,1 nohup python run_all.py > run.log 2>&1 &

(1) download data if missing  (2) per-dataset graph + embeddings, GNN baselines  (3) LLM jobs (f0 / f1 for every
neighbour strategy) per (LLM, dataset)  (4) offline routing analysis + distillation  (5) results/summary.md + figures
Options: --quick (1 LLM, 3 datasets)  --config path  --skip_download
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from needn import runtime  # noqa: E402  (stdlib-only module)

runtime.setup_env()  # VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc on the server), UTF-8, PCI_BUS_ID order, ...

REQS = [("yaml", "pyyaml"), ("numpy", "numpy"), ("sklearn", "scikit-learn"), ("scipy", "scipy"),
        ("matplotlib", "matplotlib"), ("transformers", "transformers>=4.51"),
        ("sentence_transformers", "sentence-transformers>=3.0"), ("torch_geometric", "torch_geometric"),
        ("sentencepiece", "sentencepiece"), ("google.protobuf", "protobuf")]


def check_env():
    """installs only MISSING packages (never upgrades torch; serialised across projects; see runtime.py)."""
    reqs = list(REQS)
    if os.environ.get("LLM_BACKEND", "vllm") != "hf":
        reqs.append(("vllm", "vllm>=0.8.5"))
    runtime.ensure_packages(reqs)
    import torch
    print(f"[run_all] python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}", flush=True)
    runtime.report_gpus(lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip_download", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    check_env()
    from needn import download
    from needn.experiments import build_jobs
    from needn.scheduler import run_jobs

    if not a.skip_download:
        download.main()
    summary = run_jobs(build_jobs(a.config, a.quick), ROOT / "logs", ROOT)
    subprocess.call([sys.executable, "-m", "needn.aggregate", *(["--config", a.config] if a.config else [])], cwd=ROOT)
    print(f"[run_all] finished in {(time.time() - t0) / 3600:.1f} h; failed={summary['failed']} "
          f"skipped={summary['skipped']}; see results/summary.md", flush=True)


if __name__ == "__main__":
    main()
