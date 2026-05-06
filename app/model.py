from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Swap this for your own fine-tuned model path/name when ready
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, output_attentions=True
        )
        _model.eval()


def predict(text: str) -> dict:
    _load()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    label_names = _model.config.id2label  # e.g. {0: "NEGATIVE", 1: "POSITIVE"}
    predicted_id = int(torch.argmax(outputs.logits, dim=-1))
    predicted_label = label_names[predicted_id].capitalize()

    # Average CLS-token attention across all heads in the last layer
    # Shape per layer: (batch, heads, seq_len, seq_len)
    last_attn = outputs.attentions[-1][0]       # (heads, seq_len, seq_len)
    cls_attn = last_attn[:, 0, :].mean(dim=0)   # (seq_len,)
    cls_attn = cls_attn.tolist()

    raw_tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Merge subword tokens and accumulate their attention scores
    merged = []
    for token, score in zip(raw_tokens, cls_attn):
        if token in ("[CLS]", "[SEP]", "<s>", "</s>"):
            continue
        if token.startswith("##"):
            if merged:
                merged[-1]["display"] += token[2:]
                merged[-1]["score"] += score
        else:
            merged.append({"display": token, "score": score})

    # Normalise scores 0-1
    if merged:
        lo = min(t["score"] for t in merged)
        hi = max(t["score"] for t in merged)
        span = hi - lo or 1.0
        for t in merged:
            t["normalized"] = round((t["score"] - lo) / span, 4)

    scores = {
        label_names[i].capitalize(): round(p, 4)
        for i, p in enumerate(probs)
    }

    return {
        "text": text,
        "label": predicted_label,
        "confidence": round(probs[predicted_id], 4),
        "scores": scores,
        "tokens": merged,
    }
