from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# CHOOSE WHICH MODEL TO LOAD FROM HUGGING FACE

# Options:
#   "Tanvi0212/xlmr-trigger-detection"          → English + Arabic (full)
#   "Tanvi0212/XLM-Roberta_downsampled"         → English + Arabic (downsampled)
#   "Tanvi0212/xlmr-arabic-downsampled"         → Arabic-only (downsampled)
#   "Tanvi0212/xlmr-english-downsampled"        → English-only (downsampled)


HF_MODEL_ID = "Tanvi0212/XLM-Roberta_downsampled"   # CHANGE HERE


# Loading the  tokenizer + model directly from HF Hub
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
model.eval()

label_map = {0: "Non-trigger", 1: "Trigger"}


def isTrigger(text: str):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()

    result = {
        "label": label_map[pred_label],
        "confidence": confidence
    }

    
    # RULE-BASED LAYER (My custom logic)
    

    lowered = text.lower().strip()

    negation_words = ["not", "no", "never", "without", "don't", "didn't",
                      "none", "can't", "isn't", "aren't", "won't"]

    positive_words = ["fine", "okay", "good", "well", "better", "normal",
                      "alright", "no issue", "not bad", "recovered"]

    symptom_words = ["sick", "sad", "dizzy", "pain", "hurt", "tired",
                     "depressed", "anxious", "headache"]

    # Negation + symptom → Non-trigger
    if any(n in lowered for n in negation_words) and any(s in lowered for s in symptom_words):
        result["label"] = "Non-trigger"

    # Negation + positive → Trigger
    elif any(n in lowered for n in negation_words) and any(p in lowered for p in positive_words):
        result["label"] = "Trigger"

    # Positive alone → Non-trigger
    elif any(p in lowered for p in positive_words):
        result["label"] = "Non-trigger"

    # Confidence threshold fallback
    if confidence < 0.55:
        result["label"] = "Non-trigger"

    return result
