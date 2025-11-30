from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# DYNAMIC MODEL CACHE
# We load each HF model only once the first time it is requested.
# After that, we reuse it from memory (super fast).

model_cache = {}
tokenizer_cache = {}

label_map = {0: "Non-trigger", 1: "Trigger"}


def load_model(model_id: str):
    """
    Load a Hugging Face model dynamically.
    Cached after first load -> no slowdown.
    """
    if model_id not in model_cache:
        print(f"Loading model from Hugging Face: {model_id}")

        tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
        model_cache[model_id] = AutoModelForSequenceClassification.from_pretrained(model_id)
        model_cache[model_id].eval()

        print(f" Model loaded and cached: {model_id}")

    return tokenizer_cache[model_id], model_cache[model_id]



# MAIN TRIGGER FUNCTION

def isTrigger(text: str, model_id: str):
    """
    Run prediction using dynamically selected HF model.
    Required: model_id = HuggingFace repo name (string)
    """
    tokenizer, model = load_model(model_id)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    result = {
        "label": label_map[pred_label],
        "confidence": float(confidence)
    }

    # RULE-BASED ADJUSTMENTS
   
    lowered = text.lower().strip()

    negation_words = ["not", "no", "never", "without", "don't", "didn't", "none",
                      "can't", "isn't", "aren't", "won't"]
    positive_words = ["fine", "okay", "good", "well", "better", "normal",
                      "alright", "no issue", "not bad", "recovered"]
    symptom_words = ["sick", "sad", "dizzy", "pain", "hurt", "tired",
                     "depressed", "anxious", "headache"]

    # Case 1: Negation + symptom → Non-trigger
    if any(n in lowered for n in negation_words) and any(s in lowered for s in symptom_words):
        result["label"] = "Non-trigger"

    # Case 2: Negation + positive → Trigger
    elif any(n in lowered for n in negation_words) and any(p in lowered for p in positive_words):
        result["label"] = "Trigger"

    # Case 3: Positive words alone → Non-trigger
    elif any(p in lowered for p in positive_words):
        result["label"] = "Non-trigger"

    # Optional: low confidence → assume safe
    if result["confidence"] < 0.55:
        result["label"] = "Non-trigger"

    return result
