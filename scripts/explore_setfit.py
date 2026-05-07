"""
Exploratory script — SetFit model internals for heatmap viability.

Run inside the container or a venv that has setfit + mlflow installed:
    python scripts/explore_setfit.py

Edit the two constants at the top before running.
"""
import os
import sys
import torch

# ── Config — edit these ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI       = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_SENTIMENT_MODEL_URI = os.environ.get("MLFLOW_SENTIMENT_MODEL_URI", "models:/Sentiment/Production")
SAMPLE_TEXT = "The staff were incredibly kind and the ward was spotless."
# ─────────────────────────────────────────────────────────────────────────────

import mlflow
from setfit import SetFitModel

SEP = "-" * 60


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Load model from MLflow ─────────────────────────────────────────────────
section("1. Loading model from MLflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
local_path = mlflow.artifacts.download_artifacts(MLFLOW_SENTIMENT_MODEL_URI)
print(f"Downloaded to: {local_path}")

model = SetFitModel.from_pretrained(local_path)
print(f"Model type:    {type(model)}")


# ── 2. Basic prediction ───────────────────────────────────────────────────────
section("2. Basic prediction")
pred = model.predict([SAMPLE_TEXT])
print(f"Input:      {SAMPLE_TEXT!r}")
print(f"Prediction: {pred}")


# ── 3. Model structure ────────────────────────────────────────────────────────
section("3. Model structure")
print(f"model_body type: {type(model.model_body)}")
print(f"model_head type: {type(model.model_head)}")

# SetFit body is a SentenceTransformer — its first module is the transformer
try:
    body_modules = list(model.model_body.named_children())
    print("\nBody modules:")
    for name, mod in body_modules:
        print(f"  [{name}] {type(mod).__name__}")
except Exception as e:
    print(f"Could not inspect body modules: {e}")


# ── 4. Try to get the underlying HuggingFace transformer ─────────────────────
section("4. Underlying HuggingFace transformer")
hf_model = None
try:
    # SentenceTransformer wraps a Transformer module which has .auto_model
    transformer_module = model.model_body[0]
    hf_model = transformer_module.auto_model
    print(f"HF model type:   {type(hf_model)}")
    print(f"HF model config: {hf_model.config.model_type}")
    print(f"output_attentions currently: {hf_model.config.output_attentions}")
except Exception as e:
    print(f"Could not access HF model: {e}")


# ── 5. Try attention weights ──────────────────────────────────────────────────
section("5. Attention weights from transformer body")
if hf_model is not None:
    try:
        tokenizer = transformer_module.tokenizer
        inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt", truncation=True, max_length=128)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"Tokens: {tokens}")

        hf_model.config.output_attentions = True
        with torch.no_grad():
            outputs = hf_model(**inputs)

        if outputs.attentions:
            # Strategy A: last layer only (often flat/diffuse)
            last_attn = outputs.attentions[-1]
            cls_last  = last_attn[0, :, 0, :].mean(dim=0)

            # Strategy B: average CLS attention across ALL layers (richer signal)
            all_layers = torch.stack([layer[0, :, 0, :].mean(dim=0)
                                      for layer in outputs.attentions])
            cls_all = all_layers.mean(dim=0)

            for label, scores in [("Last layer only", cls_last), ("All layers avg", cls_all)]:
                lo, hi = scores.min().item(), scores.max().item()
                span = hi - lo or 1.0
                print(f"\n{label}  (min={lo:.4f} max={hi:.4f} range={span:.4f}):")
                for tok, score in zip(tokens, scores.tolist()):
                    norm = (score - lo) / span
                    bar  = "█" * int(norm * 40)
                    print(f"  {tok:<20} {score:.4f}  {norm:.2f}  {bar}")

            print("\n✓ Attention weights ARE available — heatmap will work.")
        else:
            print("✗ No attention weights returned.")
    except Exception as e:
        print(f"Failed: {e}")
else:
    print("Skipped — no HF model found.")


# ── 6. Try predict_proba (needed for SHAP) ───────────────────────────────────
section("6. predict_proba availability (needed for SHAP)")
try:
    proba = model.predict_proba([SAMPLE_TEXT])
    print(f"predict_proba output: {proba}")
    print("✓ predict_proba available — SHAP will work.")
except AttributeError:
    print("✗ No predict_proba method — SHAP not directly usable.")
except Exception as e:
    print(f"predict_proba failed: {e}")


# ── 7. Summary ────────────────────────────────────────────────────────────────
section("Summary")
print("Options for heatmap, best → fallback:")
print("  1. Attention weights  — use CLS token attention from model_body[0].auto_model")
print("     (works if output_attentions=True is supported by the base model)")
print("  2. SHAP               — use shap.Explainer(model.predict_proba, masker)")
print("     (works if predict_proba is available, slower)")
print("  3. Token length proxy — highlight longer / rarer tokens")
print("     (no model access needed, purely cosmetic)")
