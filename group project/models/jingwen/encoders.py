"""Frozen text encoders: one vector per tweet, cached on disk."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import hashlib

from data import text_fingerprint

# name -> (hugging face id, pooling, text prefix)
ENCODERS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "mean", ""),
    "bge_large": ("BAAI/bge-large-en-v1.5", "cls", ""),
    "e5_large": ("intfloat/e5-large-v2", "mean", "query: "),
    "mxbai_large": ("mixedbread-ai/mxbai-embed-large-v1", "cls", ""),
    "bertweet": ("vinai/bertweet-base", "mean", ""),
    "twitter_roberta": ("cardiffnlp/twitter-roberta-base-sentiment-latest", "mean", ""),
    "modernbert_classification": ("nomic-ai/modernbert-embed-base", "mean", "classification: "),
    "modernbert_clustering": ("nomic-ai/modernbert-embed-base", "mean", "clustering: "),
    "gte_modernbert": ("Alibaba-NLP/gte-modernbert-base", "cls", ""),
}

# instruction-conditioned views of one instruction-tuned encoder: same tweet, different task descriptions
E5_INSTRUCT = "intfloat/multilingual-e5-large-instruct"
INSTRUCTIONS = {
    "sentiment": "Classify the sentiment of a given tweet towards the brand or game it mentions as negative, "
                 "positive, neutral, or irrelevant",
    "topic": "Identify the main topic of a given tweet",
    "emotion": "Classify the emotion expressed in the given Twitter message into one of the six emotions: "
               "anger, fear, joy, love, sadness, and surprise",
    "relevance": "Decide whether a given tweet expresses an opinion about the brand or game it mentions, "
                 "or is unrelated to it",
    "style": "Classify a given tweet as news, a personal opinion, a complaint, or casual chatter",
}
# a second wording of the three main views, to test prompt sensitivity and prompt averaging
INSTRUCTIONS_V2 = {
    "sentiment_v2": "Given a tweet that mentions a brand or a video game, classify the author's attitude toward it",
    "topic_v2": "Classify the topic of a given tweet, such as gaming, technology, business, sports, or politics",
    "style_v2": "Decide whether a given tweet is a news headline, a reposted link, or a personal post",
    "relevance_v2": "Judge whether the author of a given tweet gives a personal view of the product or game "
                    "named in it",
}
for view, instruction in (INSTRUCTIONS | INSTRUCTIONS_V2).items():
    ENCODERS[f"e5i_{view}"] = (E5_INSTRUCT, "mean", f"Instruct: {instruction}\nQuery: ")
    # the same view, mean-pooled over the last four layers instead of the last one
    ENCODERS[f"e5i_{view}_last4"] = (E5_INSTRUCT, "mean_last4", f"Instruct: {instruction}\nQuery: ")


def load_encoder(name, device="cuda"):
    model_id = ENCODERS[name][0]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode(texts, name, tokenizer, model, batch_size=32, max_length=192, device="cuda"):
    _, pooling, prefix = ENCODERS[name]
    max_length = min(max_length, tokenizer.model_max_length, model.config.max_position_embeddings - 2)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        output = model(**inputs, output_hidden_states=pooling == "mean_last4")
        hidden = torch.stack(output.hidden_states[-4:]).mean(0) if pooling == "mean_last4" else output.last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors)


def cached_encode(texts, name, cache_dir, tag="plain", max_length=192, **kwargs):
    """Encode `texts` with encoder `name`, reusing a cache keyed by the exact texts."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = hashlib.sha256(repr(ENCODERS[name]).encode("utf-8")).hexdigest()[:8]
    path = cache_dir / f"{name}_{spec}_{tag}_len{max_length}_{text_fingerprint(texts)}.npy"
    if path.exists():
        return np.load(path)
    tokenizer, model = load_encoder(name)
    vectors = encode(texts, name, tokenizer, model, max_length=max_length, **kwargs)
    np.save(path, vectors)
    del model
    torch.cuda.empty_cache()
    return vectors
