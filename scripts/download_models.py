from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print("Downloading sentiment model...")
AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

print("Downloading theme (NLI) model...")
pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

print("Models ready.")
