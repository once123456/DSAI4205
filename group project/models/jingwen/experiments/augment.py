"""Step 7a: label-preserving paraphrases of every training-pool tweet with a local LLM (gemma-3-4b-it).

Only the 591 pool tweets are rewritten; test tweets are never touched. Whether a paraphrase may be used is
decided later inside each CV fold (only paraphrases of the fold's training rows are added).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import load_splits, WORK_DIR


MODEL_DIR = os.environ.get("GEMMA_DIR", "google/gemma-3-4b-it")   # a local copy or the Hugging Face id
PROMPT = ("Rewrite the tweet below in different words. Keep the same meaning, the same attitude toward any brand, "
          "game or person it mentions, and the same topic. Keep the tweet style and about the same length. "
          "Reply with the rewritten tweet only.\n\nTweet: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    train, validation, _ = load_splits()
    pool = pd.concat([train, validation], ignore_index=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, padding_side="left")
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    rows = []
    for copy in range(args.copies):
        torch.manual_seed(copy)
        for start in range(0, len(pool), args.batch):
            part = pool.iloc[start:start + args.batch]
            chats = [[{"role": "user", "content": [{"type": "text", "text": PROMPT.format(text=t)}]}] for t in part.text]
            inputs = processor.apply_chat_template(chats, add_generation_prompt=True, tokenize=True, padding=True,
                                                   return_dict=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=96, do_sample=True, temperature=0.8, top_p=0.95)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row_id, label, text in zip(part.row_id, part.Sentiment, texts):
                rows.append({"row_id": row_id, "copy": copy, "Sentiment": label, "text": text.strip().strip('"')})
            print(f"copy {copy} {start + len(part)}/{len(pool)}", flush=True)
    out_dir = WORK_DIR / "aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "paraphrases.csv", index=False)


if __name__ == "__main__":
    main()
