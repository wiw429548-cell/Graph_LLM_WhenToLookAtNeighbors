"""Build the processed graph + node text embeddings for one dataset."""
import argparse

from .data import build
from .run_llm import embeddings
from .utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = build(a.dataset, cfg)
    embeddings(g, cfg)
    print(a.dataset, "nodes", len(g["y"]), "classes", len(g["label_names"]), "homophily", round(g["homophily"], 3))


if __name__ == "__main__":
    main()
