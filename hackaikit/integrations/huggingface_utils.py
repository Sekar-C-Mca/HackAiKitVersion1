from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForObjectDetection
# Add other AutoModel types as needed

def get_hf_pipeline(task, model_name=None, device=-1, token=None): # device -1 for CPU, 0 for CUDA
    """
    Loads a Hugging Face pipeline.
    Args:
        task (str): e.g., "sentiment-analysis", "text-generation", "object-detection"
        model_name (str, optional): Specific model from Hugging Face Hub.
        device (int, optional): -1 for CPU, >=0 for GPU.
        token (str, optional): Hugging Face API token for private models.
    Returns:
        transformers.Pipeline
    """
    try:
        if model_name:
            return pipeline(task, model=model_name, tokenizer=model_name, device=device, use_auth_token=token if token else None)
        return pipeline(task, device=device, use_auth_token=token if token else None)
    except Exception as e:
        print(f"Error loading Hugging Face pipeline for task '{task}' with model '{model_name}': {e}")
        return None

# Example of loading a specific model and tokenizer (useful for more control)
def load_hf_classification_model(model_name, num_labels=2, token=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token if token else None)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, use_auth_token=token if token else None)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading HF classification model '{model_name}': {e}")
        return None, None

def load_hf_object_detection_model(model_name, token=None):
    try:
        model = AutoModelForObjectDetection.from_pretrained(model_name, use_auth_token=token if token else None)
        # Object detection often needs a feature extractor too, not just a tokenizer
        # from transformers import AutoFeatureExtractor
        # feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # return model, feature_extractor
        print(f"Note: For HF object detection, you might also need a specific feature_extractor with the model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading HF object detection model '{model_name}': {e}")
        return None

# Add more utility functions as needed (e.g., for fine-tuning, specific model types)