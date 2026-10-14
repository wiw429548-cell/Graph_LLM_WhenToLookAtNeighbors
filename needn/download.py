"""Dataset downloader (idempotent, resumable). Honors HF_ENDPOINT (e.g. https://hf-mirror.com).

LLMNodeBed (ICML'25) PyG .pt files with raw node texts: homophilic citation / web / e-commerce / social graphs and the
heterophilic WebKB graphs; HeTGB (heterophilic text-attributed graph benchmark) Actor + Amazon-ratings text graphs.
"""
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

LLMNODEBED = ["cora", "citeseer", "pubmed", "wikics", "arxiv", "instagram", "reddit", "history", "photo", "computer",
              "cornell", "texas", "washington", "wisconsin"]
HETGB = ["Actor.npz", "Amazon.npz"]  # node_texts, node_labels, label_texts, edges


def fetch(url, dst: Path, retries=5):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = Path(str(dst) + ".part")
    for attempt in range(retries):
        try:
            pos = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={pos}-"} if pos else {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                if pos and r.status != 206:
                    pos = 0
                    part.unlink(missing_ok=True)
                with open(part, "ab") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
            part.rename(dst)
            print(f"[download] ok {dst}", flush=True)
            return dst
        except Exception as e:  # noqa
            print(f"[download] retry {attempt + 1}/{retries} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def main():
    """a failed file does not abort run_all: the prep job of that dataset fails (-> '--' in the tables) and every
    other dataset still runs"""
    files = [(f"{HF}/datasets/xxwu/LLMNodeBed/resolve/main/{d}.pt", RAW / "llmnodebed" / f"{d}.pt") for d in LLMNODEBED]
    files += [(f"{HF}/datasets/jinxi0219/HeTGB/resolve/main/{z}", RAW / "hetgb" / z) for z in HETGB]
    failed = []
    for url, dst in files:
        try:
            fetch(url, dst)
        except Exception as e:  # noqa: BLE001
            failed.append(dst.name)
            print(f"[download] giving up on {dst.name}: {e}", flush=True)
    if failed:
        print(f"[download] WARNING: missing {failed}; the corresponding datasets will be reported as '--'", flush=True)


if __name__ == "__main__":
    main()
