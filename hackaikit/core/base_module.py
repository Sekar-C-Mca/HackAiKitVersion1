from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np


class BaseModule(ABC):
    """
    Base class for all HackAI-Kit modules.
    
    This abstract class defines the common interface that all modules must implement.
    Each module specializes in a specific AI/ML domain and provides standardized methods
    for training, inference, visualization, and more.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the module.
        
        Args:
            config_manager: Configuration manager to access settings
        """
        self.config_manager = config_manager
        self.models = {}  # Store trained models
        self.model_counter = 0  # Counter for generating model IDs
        self.module_name = self.__class__.__name__
        print(f"Initializing {self.module_name}...")
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the module.
        
        Returns:
            Dictionary containing module information
        """
        return {
            "name": self.module_name,
            "description": self.__doc__ or "No description available.",
            "supported_tasks": self.get_supported_tasks(),
            "supported_algorithms": self.get_supported_algorithms()
        }
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """
        Get a list of tasks supported by the module.
        
        Returns:
            List of supported task names
        """
        pass
    
    @abstractmethod
    def get_supported_algorithms(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of supported algorithms for each task.
        
        Returns:
            Dictionary mapping task names to lists of algorithm names
        """
        pass
    
    def preprocess_data(self, data: Union[pd.DataFrame, str, List[Dict]], **kwargs) -> pd.DataFrame:
        """
        Preprocess data before processing.
        
        Args:
            data: Input data as DataFrame, CSV string, or list of dictionaries
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed data as DataFrame
        """
        # Convert string (CSV) to DataFrame
        if isinstance(data, str):
            try:
                data = pd.read_csv(data)
            except Exception as e:
                raise ValueError(f"Error parsing CSV data: {str(e)}")
        
        # Convert list of dictionaries to DataFrame
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            data = pd.DataFrame(data)
        
        # Validate that we now have a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame, CSV string, or list of dictionaries")
            
        return data
    
    def generate_model_id(self) -> str:
        """
        Generate a unique model ID.
        
        Returns:
            Unique model ID string
        """
        module_prefix = self.__class__.__name__[:3].upper()
        self.model_counter += 1
        return f"{module_prefix}_{self.model_counter}"
    
    @abstractmethod
    def train(self, data: Union[pd.DataFrame, str, List[Dict]], **kwargs) -> Dict[str, Any]:
        """
        Train a model on the provided data.
        
        Args:
            data: Training data
            **kwargs: Training parameters specific to the algorithm
            
        Returns:
            Dictionary with training results, including model_id
        """
        pass
    
    @abstractmethod
    def predict(self, model_id: str, data: Union[pd.DataFrame, str, List[Dict]], **kwargs) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model_id: ID of the trained model to use
            data: Data to make predictions on
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with prediction results
        """
        pass
    
    @abstractmethod
    def evaluate(self, model_id: str, data: Optional[Union[pd.DataFrame, str, List[Dict]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_id: ID of the trained model to evaluate
            data: Optional evaluation data (if None, uses test data from training)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def visualize(self, **kwargs) -> None:
        """
        Generate visualizations based on models, data, or results.
        
        Args:
            **kwargs: Visualization parameters specific to the module
        """
        pass
    
    def save_model(self, model_id: str, path: str) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_id: ID of the model to save
            path: Directory path to save the model
            
        Returns:
            Path to the saved model file
        """
        raise NotImplementedError("save_model not implemented for this module")
    
    def load_model(self, path: str) -> str:
        """
        Load a model from disk.
        
        Args:
            path: Path to the model file
            
        Returns:
            ID of the loaded model
        """
        raise NotImplementedError("load_model not implemented for this module")
    
    def process(self, data: Any, task: str, **kwargs) -> Dict[str, Any]:
        """
        Process data with the appropriate algorithm for the specified task.
        
        This is a convenience method that routes to the appropriate specific method
        based on the task.
        
        Args:
            data: Input data
            task: Task to perform
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with processing results
        """
        # Check if task is supported
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task '{task}' is not supported by this module. " 
                           f"Supported tasks: {self.get_supported_tasks()}")
        
        # Task-specific processing should be implemented by subclasses
        raise NotImplementedError(f"process method for task '{task}' not implemented")