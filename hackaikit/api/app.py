from flask import Flask, request, jsonify
from hackaikit.core.config_manager import ConfigManager
from hackaikit.modules.nlp import NLPModule
from hackaikit.modules.computer_vision import ComputerVisionModule
# Import other modules as you build them

# --- Global Initializations ---
# It's generally better to initialize these within an app factory if your app grows
# but for a simple kit, this can be okay.
# Ensure config.ini exists or environment variables are set for API keys.
try:
    config = ConfigManager(config_file='../config.ini') # Adjust path if running from api directory
except FileNotFoundError:
    config = ConfigManager(config_file='config.ini') # If running from root

nlp_module = NLPModule(config_manager=config)
cv_module = ComputerVisionModule(config_manager=config)
# Initialize other modules...
# supervised_module = SupervisedLearningModule(config_manager=config)
# ... etc.

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to HackAI-Kit API!"

@app.route('/api/nlp/sentiment', methods=['POST'])
def nlp_sentiment_route():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in JSON payload"}), 400
    text = request.json['text']
    provider = request.json.get('provider', 'huggingface') # Default to hf
    try:
        result = nlp_module.sentiment_analysis(text, provider=provider)
        return jsonify({"text": text, "provider": provider, "sentiment": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/nlp/generate', methods=['POST'])
def nlp_generate_route():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Missing 'prompt' in JSON payload"}), 400
    prompt = request.json['prompt']
    provider = request.json.get('provider', 'openai') # Default to openai
    model = request.json.get('model') # Optional model override
    try:
        kwargs = {}
        if model:
            kwargs['model'] = model
        result = nlp_module.generate_text(prompt, provider=provider, **kwargs)
        return jsonify({"prompt": prompt, "provider": provider, "generated_text": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cv/classify', methods=['POST'])
def cv_classify_route():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['file']
    provider = request.form.get('provider', 'huggingface')
    # You'll need to save the file temporarily or process its stream
    # For simplicity, let's assume PIL can open it from stream
    try:
        from PIL import Image
        img = Image.open(file.stream)
        result = cv_module.image_classification(img, provider=provider)
        return jsonify({"provider": provider, "classification": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cv/object_detection', methods=['POST'])
def cv_object_detection_route():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['file']
    provider = request.form.get('provider', 'huggingface')
    model = request.form.get('model') # Optional model override
    try:
        from PIL import Image
        img = Image.open(file.stream)
        kwargs = {}
        if model:
            kwargs['model'] = model
        result = cv_module.object_detection(img, provider=provider, **kwargs)
        return jsonify({"provider": provider, "detection_results": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add more routes for other modules and tasks

if __name__ == '__main__':
    # For development:
    # In production, use a proper WSGI server like Gunicorn:
    # gunicorn --bind 0.0.0.0:5000 hackaikit.api.app:app
    app.run(debug=True, host='0.0.0.0', port=5000)