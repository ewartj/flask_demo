"""
Model inference layer for the Sentiment + Theme demo.

Demo models (swappable):
  SENTIMENT_MODEL  — cardiffnlp/twitter-roberta-base-sentiment-latest  (3-class)
  ZERO_SHOT_MODEL  — cross-encoder/nli-deberta-v3-small  (zero-shot theme)

Production swap (SetFit):
  Uncomment the SetFit block below and point MODEL paths at your
  /app/ext_models/<name> directories. The returned dict shape is identical.
"""
from __future__ import annotations
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch

# ── Demo model identifiers ───────────────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "cross-encoder/nli-deberta-v3-small"

# Mirror Themiator's theme label set
THEME_LABELS = [
    "Care & service",
    "Consultation & communication",
    "Staff professionalism & attitude",
    "Environment & facilities",
    "Waiting times & access",
    "Medication & treatment",
]

# ── SetFit production swap (uncomment + set paths) ───────────────────────────
# from setfit import SetFitModel
# SENTIMENT_MODEL_PATH = "/app/ext_models/Sentiment.BEST.balanced.4epochs"
# THEME_MODEL_PATH     = "/app/ext_models/Theme.BEST.imbal.2eps"
# SENTIMENT_INT_TO_LABEL = {-1: "Negative", 0: "Neutral", 1: "Positive"}
# THEME_INT_TO_LABEL = {
#     0: "Unclassified", 1: "Care & service", 2: "Consultation & communication",
#     3: "Staff professionalism & attitude", 4: "Environment & facilities",
# }

# ── Lazy singletons ──────────────────────────────────────────────────────────
_sentiment_tok = None
_sentiment_mdl = None
_theme_pipe    = None


# ── Text cleaning (matches Themiator's clean_text pipeline) ──────────────────
def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\d+", "", text)
    return text


# ── Loaders ──────────────────────────────────────────────────────────────────
def _load_sentiment() -> None:
    global _sentiment_tok, _sentiment_mdl
    if _sentiment_mdl is None:
        _sentiment_tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        _sentiment_mdl = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, output_attentions=True
        )
        _sentiment_mdl.eval()


def _load_theme() -> None:
    global _theme_pipe
    if _theme_pipe is None:
        _theme_pipe = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=-1,
        )


# ── Token merging (handles RoBERTa Ġ, BERT ##, SentencePiece ▁) ─────────────
def _merge_tokens(tokens: list[str], scores: list[float]) -> list[dict]:
    SPECIAL = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]"}
    merged: list[dict] = []

    for tok, score in zip(tokens, scores):
        if tok in SPECIAL:
            continue
        if tok.startswith("Ġ"):              # RoBERTa: space = new word
            merged.append({"display": tok[1:], "score": score})
        elif tok.startswith("##"):           # BERT subword continuation
            if merged:
                merged[-1]["display"] += tok[2:]
                merged[-1]["score"]   += score
        elif tok.startswith("▁"):            # SentencePiece: new word
            merged.append({"display": tok[1:], "score": score})
        elif merged:                         # bare continuation
            merged[-1]["display"] += tok
            merged[-1]["score"]   += score
        else:
            merged.append({"display": tok, "score": score})

    if merged:
        lo   = min(t["score"] for t in merged)
        hi   = max(t["score"] for t in merged)
        span = hi - lo or 1.0
        for t in merged:
            t["normalized"] = round((t["score"] - lo) / span, 4)

    return merged


# ── Public API ───────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    _load_sentiment()
    _load_theme()

    clean = _clean(text)

    # — Sentiment ——————————————————————————————————————————————————————————————
    inputs = _sentiment_tok(clean, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _sentiment_mdl(**inputs)

    probs        = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    id2label     = _sentiment_mdl.config.id2label
    predicted_id = int(torch.argmax(outputs.logits))

    raw_tokens = _sentiment_tok.convert_ids_to_tokens(inputs["input_ids"][0])
    cls_attn   = outputs.attentions[-1][0][:, 0, :].mean(dim=0).tolist()
    tokens     = _merge_tokens(raw_tokens, cls_attn)

    sentiment = {
        "label":      id2label[predicted_id].capitalize(),
        "confidence": round(probs[predicted_id], 4),
        "scores":     {id2label[i].capitalize(): round(p, 4) for i, p in enumerate(probs)},
        "tokens":     tokens,
    }

    # — Theme ——————————————————————————————————————————————————————————————————
    zs = _theme_pipe(clean, THEME_LABELS, multi_label=False)
    theme = {
        "label":      zs["labels"][0],
        "confidence": round(zs["scores"][0], 4),
        "scores":     {
            label: round(score, 4)
            for label, score in zip(zs["labels"], zs["scores"])
        },
    }

    return {"text": text, "sentiment": sentiment, "theme": theme}
