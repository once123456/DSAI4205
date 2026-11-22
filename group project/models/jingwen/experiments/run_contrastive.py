"""Adapt encoders with SupCon and cache the frozen vectors of the best epoch for every split."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from contrastive import train_contrastive
from data import EXPERIMENTS_DIR, load_splits, WORK_DIR
from encoders import encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=["bge_large", "e5_large", "mpnet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--tag", default="supcon")
    args = parser.parse_args()
    train, validation, test = load_splits()
    out_dir = WORK_DIR / "cache_supcon"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.encoders:
        for seed in args.seeds:
            tokenizer, model, history = train_contrastive(
                name, train.text.tolist(), train.Sentiment.to_numpy(), validation.text.tolist(),
                validation.Sentiment.to_numpy(), epochs=args.epochs, lr=args.lr, temperature=args.temperature, seed=seed)
            stem = f"{name}_{args.tag}_seed{seed}"
            for split, frame in [("train", train), ("validation", validation), ("test", test)]:
                np.save(out_dir / f"{stem}_{split}.npy", encode(frame.text.tolist(), name, tokenizer, model))
            summary[stem] = history
            del model
            torch.cuda.empty_cache()
    path = EXPERIMENTS_DIR / f"contrastive_{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | summary, indent=2))


if __name__ == "__main__":
    main()
