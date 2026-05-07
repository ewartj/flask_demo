"""
Model inference layer.

Production (MLflow + SetFit):
  Downloads model artifacts from MLflow then loads them with SetFitModel.from_pretrained(),
  mirroring Themiator's modeller.py approach exactly.

  Required config (config.yaml or env vars):
    mlflow.tracking_uri          / MLFLOW_TRACKING_URI
    mlflow.sentiment_model_uri   / MLFLOW_SENTIMENT_MODEL_URI  (default: models:/Sentiment/Production)
    mlflow.theme_model_uri       / MLFLOW_THEME_MODEL_URI      (default: models:/Theme/Production)
    MLFLOW_TRACKING_TOKEN        — env var only, never in config

Demo (HuggingFace):
  Currently commented out — re-enable when HuggingFace is accessible.
"""
from __future__ import annotations
import re
import torch
from app.config_loader import mlflow_config, sentiment_labels, theme_labels

# ── HuggingFace demo imports — commented out while HF is blocked ──────────────
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     pipeline,
# )

# ── Config ────────────────────────────────────────────────────────────────────
_mlflow_cfg = mlflow_config()
USE_MLFLOW  = bool(_mlflow_cfg["tracking_uri"])

SENTIMENT_INT_TO_LABEL = sentiment_labels()
THEME_INT_TO_LABEL     = theme_labels()

# ── Demo model identifiers — commented out while HF is blocked ───────────────
# _DEMO_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# _DEMO_ZERO_SHOT_MODEL = "cross-encoder/nli-deberta-v3-small"

# ── Lazy singletons ──────────────────────────────────────────────────────────
_sentiment_model = None
_theme_model     = None
# _demo_sent_tok   = None   # HF — commented out while HF is blocked
# _demo_sent_mdl   = None
# _demo_theme_pipe = None


# ── Text cleaning (matches Themiator's Prediction.clean_df()) ────────────────
def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[0-9]+", "", text)
    return text


# ── MLflow + SetFit loader ────────────────────────────────────────────────────
def _load_models() -> None:
    """
    Downloads SetFit model artifacts from MLflow then loads them with
    SetFitModel.from_pretrained() — the same pattern as Themiator's run_pred_v2().
    """
    global _sentiment_model, _theme_model
    if _sentiment_model is not None:
        return

    import os
    import mlflow
    from setfit import SetFitModel

    if _mlflow_cfg["tracking_token"]:
        os.environ["MLFLOW_TRACKING_TOKEN"] = _mlflow_cfg["tracking_token"]

    mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])

    # Download artifacts locally then load with SetFit
    # URI format: models:/ModelName/Production  or  runs:/run_id/artifact_path
    sent_path  = mlflow.artifacts.download_artifacts(_mlflow_cfg["sentiment_model_uri"])
    theme_path = mlflow.artifacts.download_artifacts(_mlflow_cfg["theme_model_uri"])

    _sentiment_model = SetFitModel.from_pretrained(sent_path)
    _theme_model     = SetFitModel.from_pretrained(theme_path)

    # Enable attention on both models' RoBERTa bodies.
    # Eager implementation required — scaled_dot_product_attention doesn't
    # support output_attentions=True with this position embedding type.
    for setfit_model in (_sentiment_model, _theme_model):
        hf = setfit_model.model_body[0].auto_model
        hf.config.output_attentions    = True
        hf.config._attn_implementation = "eager"


# ── Demo loaders — commented out while HF is blocked ─────────────────────────
# def _load_demo_sentiment() -> None:
#     global _demo_sent_tok, _demo_sent_mdl
#     if _demo_sent_mdl is None:
#         _demo_sent_tok = AutoTokenizer.from_pretrained(_DEMO_SENTIMENT_MODEL)
#         _demo_sent_mdl = AutoModelForSequenceClassification.from_pretrained(
#             _DEMO_SENTIMENT_MODEL, output_attentions=True
#         )
#         _demo_sent_mdl.eval()
#
# def _load_demo_theme() -> None:
#     global _demo_theme_pipe
#     if _demo_theme_pipe is None:
#         _demo_theme_pipe = pipeline(
#             "zero-shot-classification", model=_DEMO_ZERO_SHOT_MODEL, device=-1
#         )


# ── Token merging (kept for when HF demo is re-enabled) ──────────────────────
def _merge_tokens(tokens: list[str], scores: list[float]) -> list[dict]:
    SPECIAL = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]"}
    merged: list[dict] = []
    for tok, score in zip(tokens, scores):
        if tok in SPECIAL:
            continue
        if tok.startswith("Ġ"):
            merged.append({"display": tok[1:], "score": score})
        elif tok.startswith("##"):
            if merged:
                merged[-1]["display"] += tok[2:]
                merged[-1]["score"]   += score
        elif tok.startswith("▁"):
            merged.append({"display": tok[1:], "score": score})
        elif merged:
            merged[-1]["display"] += tok
            merged[-1]["score"]   += score
        else:
            merged.append({"display": tok, "score": score})
    if merged:
        lo, hi = min(t["score"] for t in merged), max(t["score"] for t in merged)
        span = hi - lo or 1.0
        for t in merged:
            t["normalized"] = round((t["score"] - lo) / span, 4)
    return merged


# ── Attention extraction ──────────────────────────────────────────────────────
def _get_attention_tokens(setfit_model, clean: str) -> list[dict]:
    """
    Extracts per-word attention scores from a SetFit model's RoBERTa body.

    How it works:
      - Each SetFit model wraps a SentenceTransformer which wraps a RoBERTa model.
      - We pass the text to the underlying HF model with output_attentions=True.
      - Last layer attention matrix (heads × seq × seq) is averaged across heads,
        then we take the CLS row (index 0) — one score per token representing
        what the model focused on when building the sentence embedding used
        for classification.
      - Special tokens (<s>, </s>) are dropped and RoBERTa's Ġ subword pieces
        are merged before normalising scores to 0–1.
    """
    transformer_module = setfit_model.model_body[0]
    hf_model  = transformer_module.auto_model
    tokenizer = transformer_module.tokenizer

    inputs = tokenizer(clean, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = hf_model(**inputs)

    cls_attn   = outputs.attentions[-1][0, :, 0, :].mean(dim=0).tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return _merge_tokens(raw_tokens, cls_attn)


# ── Predict ───────────────────────────────────────────────────────────────────
def _predict_setfit(text: str) -> dict:
    _load_models()
    clean = _clean(text)

    sent_int      = int(_sentiment_model.predict([clean])[0])
    theme_int     = int(_theme_model.predict([clean])[0])
    sent_tokens   = _get_attention_tokens(_sentiment_model, clean)
    theme_tokens  = _get_attention_tokens(_theme_model, clean)

    return {
        "text": text,
        "sentiment": {
            "label":      SENTIMENT_INT_TO_LABEL.get(sent_int, str(sent_int)),
            "confidence": None,
            "scores":     {},
            "tokens":     sent_tokens,
        },
        "theme": {
            "label":      THEME_INT_TO_LABEL.get(theme_int, str(theme_int)),
            "confidence": None,
            "scores":     {},
            "tokens":     theme_tokens,
        },
        "source": "mlflow",
    }


# ── Demo predict — commented out while HF is blocked ─────────────────────────
# def _predict_demo(text: str) -> dict:
#     _load_demo_sentiment()
#     _load_demo_theme()
#     clean = _clean(text)
#     inputs = _demo_sent_tok(clean, return_tensors="pt", truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = _demo_sent_mdl(**inputs)
#     probs        = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
#     id2label     = _demo_sent_mdl.config.id2label
#     predicted_id = int(torch.argmax(outputs.logits))
#     raw_tokens   = _demo_sent_tok.convert_ids_to_tokens(inputs["input_ids"][0])
#     cls_attn     = outputs.attentions[-1][0][:, 0, :].mean(dim=0).tolist()
#     tokens       = _merge_tokens(raw_tokens, cls_attn)
#     from app.config_loader import theme_candidates
#     zs           = _demo_theme_pipe(clean, theme_candidates(), multi_label=False)
#     return {
#         "text": text,
#         "sentiment": {
#             "label":      id2label[predicted_id].capitalize(),
#             "confidence": round(probs[predicted_id], 4),
#             "scores":     {id2label[i].capitalize(): round(p, 4) for i, p in enumerate(probs)},
#             "tokens":     tokens,
#         },
#         "theme": {
#             "label":      zs["labels"][0],
#             "confidence": round(zs["scores"][0], 4),
#             "scores":     {l: round(s, 4) for l, s in zip(zs["labels"], zs["scores"])},
#         },
#         "source": "demo",
#     }


# ── Public API ────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    if not USE_MLFLOW:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not configured. Set it in config.yaml or as "
            "the MLFLOW_TRACKING_URI environment variable."
        )
    return _predict_setfit(text)
    # Restore this when HF is accessible:
    # return _predict_setfit(text) if USE_MLFLOW else _predict_demo(text)
