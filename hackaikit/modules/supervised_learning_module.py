import os
import time
import pickle
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Import XGBoost if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from hackaikit.core.base_module import BaseModule


class SupervisedLearningModule(BaseModule):
    """
    Supervised Learning Module for HackAI-Kit.
    
    This module provides classification and regression algorithms
    including Random Forest, Logistic/Linear Regression, SVM,
    Decision Tree, and XGBoost (if available).
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the Supervised Learning module.
        
        Args:
            config_manager: Configuration manager
        """
        super().__init__(config_manager)
        
        # Initialize storage for trained models and metadata
        self.models = {}
        self.model_data = {}
        
        # Define available algorithms
        self.algorithms = {
            "classification": {
                "random_forest": RandomForestClassifier,
                "logistic": LogisticRegression,
                "svm": SVC,
                "decision_tree": DecisionTreeClassifier,
            },
            "regression": {
                "random_forest": RandomForestRegressor,
                "linear": LinearRegression,
                "svm": SVR,
                "decision_tree": DecisionTreeRegressor,
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.algorithms["classification"]["xgboost"] = xgb.XGBClassifier
            self.algorithms["regression"]["xgboost"] = xgb.XGBRegressor
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get a list of tasks supported by the module.
        
        Returns:
            List of supported task names
        """
        return ["classification", "regression"]
    
    def get_supported_algorithms(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of supported algorithms for each task.
        
        Returns:
            Dictionary mapping task names to lists of algorithm names
        """
        return {
            "classification": list(self.algorithms["classification"].keys()),
            "regression": list(self.algorithms["regression"].keys())
        }
    
    def _determine_problem_type(self, y) -> str:
        """
        Determine whether this is a classification or regression problem.
        
        Args:
            y: Target variable
            
        Returns:
            Problem type: 'classification' or 'regression'
        """
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            return "classification"
        
        # If there are very few unique values relative to the number of samples
        # and all are integers, likely a classification problem
        unique_ratio = len(y.unique()) / len(y)
        if unique_ratio < 0.05 and all(y.apply(lambda x: float(x).is_integer())):
            return "classification"
            
        return "regression"
    
    def train(self, data: Union[pd.DataFrame, str, List[Dict]], target_column: str, 
              feature_columns: Optional[List[str]] = None, algorithm: str = "random_forest",
              problem_type: Optional[str] = None, test_size: float = 0.2, 
              random_state: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Train a supervised learning model.
        
        Args:
            data: Input data as DataFrame, CSV string, or list of dictionaries
            target_column: Name of the target column
            feature_columns: List of feature column names (if None, use all non-target columns)
            algorithm: Algorithm to use (random_forest, logistic/linear, svm, decision_tree, xgboost)
            problem_type: Type of problem ('classification' or 'regression'). If None, will be determined automatically.
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Dictionary with training results and model ID
        """
        start_time = time.time()
        
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Identify features and target
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Check if all feature columns exist
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns {missing_cols} not found in data")
            
        # Split features and target
        X = data[feature_columns]
        y = data[target_column]
        
        # Determine problem type if not provided
        if problem_type is None:
            problem_type = self._determine_problem_type(y)
            
        # Validate problem type
        if problem_type not in ["classification", "regression"]:
            raise ValueError(f"Problem type '{problem_type}' not supported. " 
                            f"Supported types: {['classification', 'regression']}")
        
        # Validate algorithm
        if algorithm not in self.algorithms[problem_type]:
            raise ValueError(f"Algorithm '{algorithm}' not supported for {problem_type}. " 
                            f"Supported algorithms: {list(self.algorithms[problem_type].keys())}")
            
        # Convert target to numeric for classification
        encoder = None
        if problem_type == "classification" and (y.dtype == 'object' or pd.api.types.is_categorical_dtype(y)):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features (optional, but recommended for some algorithms)
        if algorithm in ["svm", "logistic", "linear"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = None
            
        # Create model instance with parameters
        model_class = self.algorithms[problem_type][algorithm]
        model = model_class(random_state=random_state, **kwargs)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, problem_type)
        
        # Generate model ID
        model_id = self.generate_model_id()
        
        # Store model and metadata
        self.models[model_id] = model
        self.model_data[model_id] = {
            "problem_type": problem_type,
            "algorithm": algorithm,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "metrics": metrics,
            "encoder": encoder,
            "scaler": scaler,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "train_time": time.time() - start_time
        }
        
        # Return results
        return {
            "model_id": model_id,
            "problem_type": problem_type,
            "algorithm": algorithm,
            "metrics": metrics,
            "elapsed_time": time.time() - start_time
        }
    
    def predict(self, model_id: str, data: Union[pd.DataFrame, str, List[Dict]], 
                include_confidence: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model_id: ID of the trained model to use
            data: Data to make predictions on
            include_confidence: Whether to include confidence scores (for classification)
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model exists
        if model_id not in self.models:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        # Get model and metadata
        model = self.models[model_id]
        metadata = self.model_data[model_id]
        
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Extract features
        features = metadata["feature_columns"]
        
        # Check if all required features are present
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
            
        X = data[features]
        
        # Apply scaling if used during training
        if metadata["scaler"] is not None:
            X = metadata["scaler"].transform(X)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get confidence scores for classification
        confidence = None
        if include_confidence and metadata["problem_type"] == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                # Get maximum probability for each prediction
                confidence = np.max(proba, axis=1).tolist()
        
        # Decode labels for classification
        if metadata["encoder"] is not None:
            y_pred = metadata["encoder"].inverse_transform(y_pred)
        
        # Convert predictions to list
        predictions = y_pred.tolist()
        
        return {
            "predictions": predictions,
            "model_id": model_id,
            "confidence": confidence
        }
    
    def evaluate(self, model_id: str, data: Optional[Union[pd.DataFrame, str, List[Dict]]] = None, 
                 target_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_id: ID of the trained model to evaluate
            data: Optional evaluation data (if None, uses test data from training)
            target_column: Name of the target column in data (required if data is provided)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if model exists
        if model_id not in self.models:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        # Get model and metadata
        model = self.models[model_id]
        metadata = self.model_data[model_id]
        
        if data is None:
            # Use test data from training
            X_test = metadata["X_test"]
            y_test = metadata["y_test"]
            y_pred = metadata["y_pred"]
        else:
            # Preprocess new data
            data = self.preprocess_data(data)
            
            # Check if target column is provided
            if target_column is None:
                target_column = metadata["target_column"]
                
            # Check if target column exists
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
                
            # Get features and target
            X = data[metadata["feature_columns"]]
            y = data[target_column]
            
            # Apply scaling if used during training
            if metadata["scaler"] is not None:
                X = metadata["scaler"].transform(X)
                
            # Encode target if needed
            if metadata["encoder"] is not None:
                y = metadata["encoder"].transform(y)
                
            # Make predictions
            y_pred = model.predict(X)
            X_test = X
            y_test = y
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, metadata["problem_type"])
        
        return {
            "model_id": model_id,
            "problem_type": metadata["problem_type"],
            "algorithm": metadata["algorithm"],
            "metrics": metrics
        }
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """
        Calculate performance metrics based on problem type.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            problem_type: Type of problem ('classification' or 'regression')
            
        Returns:
            Dictionary of metrics
        """
        if problem_type == "classification":
            # Calculate classification metrics
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        else:
            # Calculate regression metrics
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))
            }
            
        return metrics
    
    def visualize(self, model_id: str, visualization_type: str = "feature_importance",
                  top_n: int = 10, **kwargs) -> None:
        """
        Generate visualizations for a trained model.
        
        Args:
            model_id: ID of the trained model
            visualization_type: Type of visualization (feature_importance, confusion_matrix, etc.)
            top_n: Number of top features to show in feature importance plot
            **kwargs: Additional visualization parameters
        """
        # Check if model exists
        if model_id not in self.models:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        # Get model and metadata
        model = self.models[model_id]
        metadata = self.model_data[model_id]
        
        if visualization_type == "feature_importance":
            self._visualize_feature_importance(model, metadata, top_n)
        elif visualization_type == "confusion_matrix":
            if metadata["problem_type"] != "classification":
                raise ValueError("Confusion matrix visualization is only available for classification problems")
            self._visualize_confusion_matrix(metadata)
        elif visualization_type == "residuals":
            if metadata["problem_type"] != "regression":
                raise ValueError("Residuals visualization is only available for regression problems")
            self._visualize_residuals(metadata)
        else:
            raise ValueError(f"Visualization type '{visualization_type}' not supported")
    
    def _visualize_feature_importance(self, model, metadata: Dict, top_n: int) -> None:
        """
        Visualize feature importance.
        
        Args:
            model: Trained model
            metadata: Model metadata
            top_n: Number of top features to show
        """
        plt.figure(figsize=(10, 6))
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Get feature names
            feature_names = metadata["feature_columns"]
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Limit to top N features
            if len(feature_importance) > top_n:
                feature_importance = feature_importance.iloc[:top_n]
                
            # Create bar plot
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f"Top {top_n} Feature Importance")
            plt.tight_layout()
            plt.show()
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = model.coef_
            
            # Handle multi-class classification
            if len(coefficients.shape) > 1:
                coefficients = np.abs(coefficients).mean(axis=0)
                
            # Get feature names
            feature_names = metadata["feature_columns"]
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            feature_importance['AbsCoefficient'] = np.abs(feature_importance['Coefficient'])
            feature_importance = feature_importance.sort_values('AbsCoefficient', ascending=False)
            
            # Limit to top N features
            if len(feature_importance) > top_n:
                feature_importance = feature_importance.iloc[:top_n]
                
            # Create bar plot
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
            plt.title(f"Top {top_n} Feature Coefficients")
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance visualization not available for this model type")
    
    def _visualize_confusion_matrix(self, metadata: Dict) -> None:
        """
        Visualize confusion matrix for classification problems.
        
        Args:
            metadata: Model metadata
        """
        plt.figure(figsize=(8, 6))
        
        # Get actual and predicted values
        y_true = metadata["y_test"]
        y_pred = metadata["y_pred"]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print(classification_report(y_true, y_pred))
    
    def _visualize_residuals(self, metadata: Dict) -> None:
        """
        Visualize residuals for regression problems.
        
        Args:
            metadata: Model metadata
        """
        # Get actual and predicted values
        y_true = metadata["y_test"]
        y_pred = metadata["y_pred"]
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot residuals vs predicted values
        ax1.scatter(y_pred, residuals)
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        
        # Plot residuals distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_id: str, path: str) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_id: ID of the model to save
            path: Directory path to save the model
            
        Returns:
            Path to the saved model file
        """
        # Check if model exists
        if model_id not in self.models:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Get model and metadata
        model = self.models[model_id]
        metadata = self.model_data[model_id]
        
        # Save model file
        model_file = os.path.join(path, f"{model_id}_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metadata file
        metadata_file = os.path.join(path, f"{model_id}_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
            
        return model_file
    
    def load_model(self, path: str) -> str:
        """
        Load a model from disk.
        
        Args:
            path: Path to the model file
            
        Returns:
            ID of the loaded model
        """
        # Get model ID from filename
        model_id = os.path.basename(path).split('_')[0]
        
        # Load model
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        # Load metadata
        metadata_file = path.replace('_model.pkl', '_metadata.pkl')
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            
        # Store model and metadata
        self.models[model_id] = model
        self.model_data[model_id] = metadata
        
        return model_id
    
    def process(self, data: Union[pd.DataFrame, str, List[Dict]], task: str, 
                target_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process data with the appropriate algorithm for the specified task.
        
        Args:
            data: Input data
            task: Task to perform ('classification' or 'regression')
            target_column: Name of the target column (required for training)
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with processing results
        """
        # Validate task
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task '{task}' not supported. " 
                           f"Supported tasks: {self.get_supported_tasks()}")
            
        # Determine what operation to perform
        operation = kwargs.pop("operation", "train")
        
        if operation == "train":
            if target_column is None:
                raise ValueError("Target column must be specified for training")
                
            return self.train(data=data, target_column=target_column, problem_type=task, **kwargs)
            
        elif operation == "predict":
            model_id = kwargs.pop("model_id", None)
            if model_id is None:
                raise ValueError("Model ID must be specified for prediction")
                
            return self.predict(model_id=model_id, data=data, **kwargs)
            
        elif operation == "evaluate":
            model_id = kwargs.pop("model_id", None)
            if model_id is None:
                raise ValueError("Model ID must be specified for evaluation")
                
            return self.evaluate(model_id=model_id, data=data, target_column=target_column, **kwargs)
            
        else:
            raise ValueError(f"Operation '{operation}' not supported. " 
                           f"Supported operations: ['train', 'predict', 'evaluate']") 