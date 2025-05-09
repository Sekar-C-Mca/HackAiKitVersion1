# HackAI-Kit: The Ultimate Plug-and-Play AI/ML Toolkit for Hackathons

HackAI-Kit is an all-in-one, modular AI/ML system designed to empower developers, data scientists, and enthusiasts to quickly build AI solutions for hackathon challenges.

## Overview

HackAI-Kit provides pre-built modules and seamless integrations with cutting-edge AI models from Hugging Face, Google Gemini, OpenAI (ChatGPT), and supports other external APIs. It aims to simplify prototyping and deployment across various AI/ML domains.

## Features (Planned & In-Progress)

* **Supervised Learning Module:** Classification/Regression (Decision Trees, RF, XGBoost, LR, SVM).
* **Unsupervised Learning Module:** Clustering (K-means, DBSCAN).
* **Computer Vision Module:** Object detection, image classification, segmentation (OpenCV, YOLO, HF Vision, Gemini Vision).
* **Natural Language Processing Module:** Text generation, summarization, translation, sentiment analysis (HF Transformers, GPT, BERT, T5, Gemini, ChatGPT).
* **Time Series Forecasting Module:** AutoARIMA, Prophet, LSTM.
* **Reinforcement Learning Module:** Q-learning, DQN, PPO with Gym integration.
* **Deep Learning Module:** Build CNNs, RNNs, GANs (TensorFlow/PyTorch, HF Transformers).
* **Large Language Models (LLM-Powered Tools):** Chatbots, fine-tuning support.
* **Sensor & Signal Data Module:** IoT anomaly detection.
* **TinyML Module:** Model quantization and optimization for edge devices.

## Key API Integrations

* **Hugging Face:** Access Model Hub, transformers, datasets, tokenizers.
* **Google Gemini:** Multimodal capabilities (text, image, etc.).
* **OpenAI (ChatGPT):** Conversational AI, text generation.
* **Custom APIs:** Extensible for other AI services.

## Setup

### 1. Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* (Optional but Recommended) Git

### 2. Clone the Repository (if you have it in git)

```bash
git clone <your-repo-url-here>
cd HackAI-Kit
```

## Installation

### Regular Python Environment

```bash
pip install hackaikit
```

For specific modules:

```bash
# Install with all dependencies
pip install hackaikit[all]

# Install specific modules
pip install hackaikit[supervised,unsupervised,llm]
```

### Google Colab Installation

```python
!pip install git+https://github.com/user/hackaikit.git
```

Or to install specific modules:

```python
!pip install git+https://github.com/user/hackaikit.git#egg=hackaikit[supervised,llm]
```

## Quick Start Guide for Google Colab

```python
# Install HackAI-Kit
!pip install git+https://github.com/user/hackaikit.git

# Import necessary modules
from hackaikit.core.module_manager import ModuleManager
from hackaikit.core.config_manager import ConfigManager

# Initialize the managers
config_manager = ConfigManager()
module_manager = ModuleManager(config_manager=config_manager)

# List available modules
available_modules = module_manager.discover_modules()
print(f"Available modules: {available_modules}")
```

## Module Usage Guide

### 1. Supervised Learning Module

Use this module for classification and regression tasks with algorithms like Random Forest, XGBoost, SVM, etc.

```python
# Get the Supervised Learning module
supervised_module = module_manager.get_module("SupervisedLearningModule")

# Load data (example with pandas)
import pandas as pd
data = pd.read_csv('your_data.csv')

# Train a model
model_info = supervised_module.train(
    data=data,
    target_column='target',
    algorithm='random_forest',
    test_size=0.2,
    random_state=42
)

# Make predictions
predictions = supervised_module.predict(
    model_id=model_info['model_id'],
    data=test_data
)

# Evaluate the model
metrics = supervised_module.evaluate(model_id=model_info['model_id'])
print(f"Model metrics: {metrics}")
```

### 2. Unsupervised Learning Module

Use this module for clustering and dimensionality reduction.

```python
# Get the Unsupervised Learning module
unsupervised_module = module_manager.get_module("UnsupervisedLearningModule")

# Perform clustering
cluster_results = unsupervised_module.process(
    data=data,
    task='clustering',
    algorithm='kmeans',
    n_clusters=5
)

# Visualize clusters
unsupervised_module.visualize(
    data=data,
    labels=cluster_results['cluster_labels'],
    method='2d_scatter'
)
```

### 3. Time Series Module

Use this module for time series forecasting using ARIMA, SARIMA, Prophet, and LSTM models.

```python
# Get the Time Series module
timeseries_module = module_manager.get_module("TimeSeriesModule")

# Load time series data
data = pd.read_csv('time_series_data.csv', parse_dates=['timestamp'])

# Perform forecasting
forecast = timeseries_module.process(
    data=data,
    task='forecasting',
    algorithm='prophet',
    timestamp_column='timestamp',
    value_column='value',
    steps=30  # forecast 30 steps ahead
)

# Visualize the forecast
timeseries_module.visualize(
    data=data,
    forecast=forecast,
    timestamp_column='timestamp',
    value_column='value'
)
```

### 4. Deep Learning Module

Use this module for neural networks, including classifiers, CNNs, and autoencoders.

```python
# Get the Deep Learning module
dl_module = module_manager.get_module("DeepLearningModule")

# Train a neural network
model_info = dl_module.train(
    data=data,
    task='classification',
    architecture='cnn',
    target_column='label',
    feature_columns=['feature1', 'feature2'],
    epochs=10,
    batch_size=32
)

# Make predictions
predictions = dl_module.predict(
    model_id=model_info['model_id'],
    data=test_data
)
```

### 5. Reinforcement Learning Module

Use this module for Q-learning, DQN, and policy gradient methods.

```python
# Get the RL module
rl_module = module_manager.get_module("RLModule")

# Train an RL agent
agent_info = rl_module.train(
    env_name='CartPole-v1',
    algorithm='dqn',
    episodes=500
)

# Test the agent
rewards = rl_module.test(agent_id=agent_info['agent_id'], episodes=10)
print(f"Average reward: {sum(rewards)/len(rewards)}")
```

### 6. LLM Tools Module

Use this module to interact with large language models like OpenAI GPT, Google Gemini, and Hugging Face models.

```python
# Get the LLM Tools module
llm_module = module_manager.get_module("LLMToolsModule")

# Chat with an LLM
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = llm_module.process(
    data=messages,
    task="chat",
    provider="openai",
    model="gpt-3.5-turbo"
)

print(f"Response: {response['response']}")

# Generate text
generated_text = llm_module.process(
    data="Write a poem about artificial intelligence",
    task="text_generation",
    provider="gemini",
    model="gemini-pro"
)

print(f"Generated text: {generated_text['generated_text']}")
```

### 7. Sensor & Signal Module

Use this module for IoT data processing and anomaly detection.

```python
# Get the Sensor & Signal module
sensor_module = module_manager.get_module("SensorModule")

# Detect anomalies in sensor data
anomalies = sensor_module.process(
    data=sensor_data,
    task='anomaly_detection',
    algorithm='isolation_forest',
    timestamp_column='timestamp',
    feature_columns=['sensor1', 'sensor2', 'sensor3']
)

# Visualize anomalies
sensor_module.visualize(
    data=sensor_data,
    anomalies=anomalies['anomaly_indices'],
    timestamp_column='timestamp'
)
```

### 8. TinyML Module

Use this module for model optimization and deployment on edge devices.

```python
# Get the TinyML module
tinyml_module = module_manager.get_module("TinyMLModule")

# Optimize a model for edge deployment
optimized_model = tinyml_module.optimize(
    model_path='my_model.h5',
    target_platform='microcontroller',
    quantization=True
)

# Convert to C header file for Arduino
c_code = tinyml_module.convert_to_c_array(
    model_path=optimized_model['model_path'],
    output_path='model_data.h'
)
```

## Using the API in Google Colab

You can also run the FastAPI server directly in Google Colab for integrating with frontend applications:

```python
# Install dependencies
!pip install git+https://github.com/user/hackaikit.git#egg=hackaikit[all]
!pip install pyngrok

# Import modules
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from hackaikit.api.main import app

# Apply nest_asyncio to allow running the server in a notebook
nest_asyncio.apply()

# Start ngrok tunnel
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

# Run the FastAPI server
uvicorn.run(app, port=8000)
```

## Environment Variables

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

In Google Colab, set environment variables directly:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_key"
os.environ["GOOGLE_API_KEY"] = "your_gemini_key"
os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_key"
```

## More Examples

Check out example notebooks in the `examples/notebooks/` directory for more detailed tutorials and use cases.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
