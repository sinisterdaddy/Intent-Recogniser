from transformers import pipeline

# Load your models (can be switched dynamically)
# Load all models
distilbert_pipeline = pipeline("text-classification", model="./distilbert", tokenizer="./distilbert")
roberta_pipeline    = pipeline("text-classification", model="./roberta", tokenizer="./roberta")
zero_shot_pipeline  = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def predict_intent(text, model_name="distilbert"):

    if model_name == "roberta":
        return roberta_pipeline(text)[0]['label']
    elif model_name == "zero_shot":
        labels = ["book_flight", "cancel_flight", "greeting", "weather", "play_music"]
        result = zero_shot_pipeline(text, candidate_labels=labels)
        return result['labels'][0]
    else:
        return distilbert_pipeline(text)[0]['label']
