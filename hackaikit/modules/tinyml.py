from hackaikit.core.base_module import BaseModule
import numpy as np
import os
import json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tempfile
import logging

logger = logging.getLogger(__name__)

class TinyMLModule(BaseModule):
    """
    Module for TinyML - deploying ML models on resource-constrained devices.
    Supports model quantization and conversion to TFLite for edge deployment.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.model = None
        self.quantized_model = None
        self.input_shape = None
        self.output_shape = None
        self.scaler = None
        self.class_labels = None
        
    def process(self, data, task="quantize", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "quantize":
            return self.quantize_model(data, **kwargs)
        elif task == "convert":
            return self.convert_to_tflite(data, **kwargs)
        elif task == "optimize":
            return self.optimize_for_edge(data, **kwargs)
        elif task == "benchmark":
            return self.benchmark_model(data, **kwargs)
        else:
            return f"TinyML task '{task}' not supported."
    
    def load_keras_model(self, model_path, input_shape=None, class_labels=None):
        """
        Load a Keras model for TinyML conversion
        
        Args:
            model_path (str): Path to the Keras model (.h5 or SavedModel)
            input_shape (tuple): Input shape of the model
            class_labels (list): List of class labels for classification models
            
        Returns:
            str: Success message or error
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.input_shape = input_shape or self.model.input_shape[1:]
            
            # Store class labels if provided
            if class_labels:
                self.class_labels = class_labels
                
            return f"Model loaded successfully from {model_path}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def quantize_model(self, input_data=None, quantization_type="int8", **kwargs):
        """
        Quantize model for smaller size and faster inference
        
        Args:
            input_data (np.ndarray): Representative input data for quantization
            quantization_type (str): Type of quantization (int8, float16, dynamic)
            
        Returns:
            dict: Quantization results
        """
        if self.model is None:
            return "No model loaded. Please load a model first."
            
        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Set quantization parameters based on type
            if quantization_type == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                # Create representative dataset from input data
                if input_data is not None:
                    def _representative_dataset():
                        for i in range(min(100, len(input_data))):
                            sample = input_data[i:i+1].astype(np.float32)
                            yield [sample]
                    
                    converter.representative_dataset = _representative_dataset
                else:
                    return "For int8 quantization, input_data is required."
                    
            elif quantization_type == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
            elif quantization_type == "dynamic":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            else:
                return f"Quantization type '{quantization_type}' not supported."
                
            # Convert model
            self.quantized_model = converter.convert()
            
            # Calculate size reduction
            original_size = self._get_model_size(self.model)
            quantized_size = len(self.quantized_model) / 1024  # in KB
            
            size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            
            return {
                "quantization_type": quantization_type,
                "original_size_kb": original_size,
                "quantized_size_kb": quantized_size,
                "size_reduction_percent": size_reduction
            }
            
        except Exception as e:
            return f"Error during quantization: {str(e)}"
    
    def convert_to_tflite(self, output_path=None, **kwargs):
        """
        Convert and save model to TFLite format
        
        Args:
            output_path (str): Path to save the TFLite model
            
        Returns:
            str: Path to the saved model or error message
        """
        if self.quantized_model is None and self.model is None:
            return "No model loaded or quantized. Please load or quantize a model first."
            
        try:
            # If model is not quantized yet, quantize with default settings
            if self.quantized_model is None:
                result = self.quantize_model(quantization_type="dynamic")
                if isinstance(result, str):  # Error message
                    return result
                    
            # Generate default output path if not provided
            if output_path is None:
                output_path = "model_tflite.tflite"
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(self.quantized_model)
                
            # Record model metadata
            metadata = {
                "input_shape": [dim if dim is not None else -1 for dim in self.input_shape],
                "output_shape": [dim if dim is not None else -1 for dim in (self.output_shape or self.model.output_shape[1:])],
                "class_labels": self.class_labels
            }
            
            # Save metadata
            metadata_path = f"{os.path.splitext(output_path)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            return {
                "tflite_model_path": output_path,
                "metadata_path": metadata_path,
                "model_size_kb": os.path.getsize(output_path) / 1024
            }
            
        except Exception as e:
            return f"Error converting to TFLite: {str(e)}"
    
    def optimize_for_edge(self, target_device="microcontroller", input_data=None, **kwargs):
        """
        Optimize model specifically for edge devices
        
        Args:
            target_device (str): Target device (microcontroller, mobile, raspberrypi)
            input_data (np.ndarray): Representative input data
            
        Returns:
            dict: Optimization results
        """
        if self.model is None:
            return "No model loaded. Please load a model first."
            
        try:
            # Set optimization parameters based on target device
            if target_device == "microcontroller":
                # Aggressive quantization for very small devices
                return self.quantize_model(
                    input_data=input_data,
                    quantization_type="int8",
                    **kwargs
                )
                
            elif target_device == "mobile":
                # Use float16 for mobile devices with more memory
                return self.quantize_model(
                    quantization_type="float16",
                    **kwargs
                )
                
            elif target_device == "raspberrypi":
                # Balanced approach for Raspberry Pi
                return self.quantize_model(
                    input_data=input_data,
                    quantization_type="dynamic",
                    **kwargs
                )
                
            else:
                return f"Target device '{target_device}' not supported."
                
        except Exception as e:
            return f"Error optimizing for edge: {str(e)}"
    
    def benchmark_model(self, input_data, num_runs=10, **kwargs):
        """
        Benchmark the model performance on representative data
        
        Args:
            input_data (np.ndarray): Input data for benchmarking
            num_runs (int): Number of benchmark runs
            
        Returns:
            dict: Benchmark results
        """
        if self.model is None and self.quantized_model is None:
            return "No model loaded. Please load a model first."
            
        try:
            # Prepare input data
            if not isinstance(input_data, np.ndarray):
                return "Input data should be a numpy array."
                
            # Benchmark original model if available
            original_times = []
            if self.model is not None:
                # Convert to TF tensor
                tf_input = tf.convert_to_tensor(input_data, dtype=tf.float32)
                
                # Warmup run
                _ = self.model(tf_input)
                
                # Benchmark runs
                for _ in range(num_runs):
                    start_time = tf.timestamp()
                    _ = self.model(tf_input)
                    end_time = tf.timestamp()
                    original_times.append((end_time - start_time).numpy())
            
            # Benchmark quantized model if available
            quantized_times = []
            if self.quantized_model is not None:
                # Create TFLite interpreter
                interpreter = tf.lite.Interpreter(model_content=self.quantized_model)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Prepare input data for TFLite
                tflite_input = input_data.astype(np.float32)
                
                # Warmup run
                interpreter.set_tensor(input_details[0]['index'], tflite_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                
                # Benchmark runs
                for _ in range(num_runs):
                    start_time = tf.timestamp()
                    interpreter.set_tensor(input_details[0]['index'], tflite_input)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]['index'])
                    end_time = tf.timestamp()
                    quantized_times.append((end_time - start_time).numpy())
            
            # Calculate statistics
            results = {}
            
            if original_times:
                avg_original = np.mean(original_times) * 1000  # Convert to ms
                std_original = np.std(original_times) * 1000
                
                results["original_model"] = {
                    "average_ms": avg_original,
                    "std_dev_ms": std_original,
                    "min_ms": np.min(original_times) * 1000,
                    "max_ms": np.max(original_times) * 1000
                }
                
            if quantized_times:
                avg_quantized = np.mean(quantized_times) * 1000  # Convert to ms
                std_quantized = np.std(quantized_times) * 1000
                
                results["quantized_model"] = {
                    "average_ms": avg_quantized,
                    "std_dev_ms": std_quantized,
                    "min_ms": np.min(quantized_times) * 1000,
                    "max_ms": np.max(quantized_times) * 1000
                }
                
                # Calculate speedup if both models were benchmarked
                if original_times:
                    speedup = avg_original / avg_quantized if avg_quantized > 0 else 0
                    results["speedup"] = speedup
            
            return results
            
        except Exception as e:
            return f"Error during benchmarking: {str(e)}"
    
    def _get_model_size(self, model):
        """Get model size in KB"""
        try:
            # Save model to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_model.h5")
            
            model.save(temp_file)
            size_kb = os.path.getsize(temp_file) / 1024
            
            # Clean up
            os.remove(temp_file)
            os.rmdir(temp_dir)
            
            return size_kb
        except:
            # If saving fails, use a rough estimation
            try:
                return sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables) * 4 / 1024  # Assuming float32 (4 bytes)
            except:
                return 0
