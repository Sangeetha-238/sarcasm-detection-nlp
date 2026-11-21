from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert-sarcasm-final"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_sarcasm(text):
    """
    Predict whether a headline is sarcastic.

    Returns:
        label: Sarcastic or Not Sarcastic
        confidence: prediction confidence between 0 and 1
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    label = (
        "Sarcastic"
        if predicted_class == 1
        else "Not Sarcastic"
    )

    return {
        "text": text,
        "prediction": label,
        "confidence": confidence
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: python src/predict.py '
            '"Enter a headline here"'
        )
        sys.exit(1)

    headline = " ".join(sys.argv[1:])

    result = predict_sarcasm(headline)

    print("\nSarcasm Detection Result")
    print("------------------------")
    print(f"Text:       {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    