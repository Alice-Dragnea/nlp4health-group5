
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "trigger_xlmr_model"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "Non-trigger", 1: "Trigger"}

def isTrigger(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    result = {"label": label_map[pred_label], "confidence": confidence}
    lowered = text.lower()

    negation_words = ["not", "no", "never", "without", "don't", "didn't", "none"]
    positive_words = ["fine", "okay", "good", "well", "better", "normal", "alright", "no issue", "not bad", "recovered"]
    symptom_words = ["sick", "sad", "dizzy", "pain", "hurt", "tired", "depressed", "anxious", "headache"]

    # Case 1: Negation + symptom → Non-trigger
    if any(n in lowered for n in negation_words) and any(s in lowered for s in symptom_words):
        if result["label"] == "Trigger":
            result["label"] = "Non-trigger"

    # Case 2: Negation + positive → Trigger (reverse of earlier mistake)
    elif any(n in lowered for n in negation_words) and any(p in lowered for p in positive_words):
        if result["label"] == "Non-trigger":
            result["label"] = "Trigger"

    # Case 3: Positive words alone → Non-trigger
    elif any(p in lowered for p in positive_words):
        if result["label"] == "Trigger":
            result["label"] = "Non-trigger"

    return result
