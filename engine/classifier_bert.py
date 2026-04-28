import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./engine/Model/clinicalbert_finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

def classify(text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=1).item())
