from hackaikit.core.base_module import BaseModule
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import joblib
import os
import xgboost as xgb

class SupervisedLearningModule(BaseModule):
    """
    Module for supervised learning tasks including classification and regression.
    Supports scikit-learn models and integrates with XGBoost.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.model = None
        self.model_type = None
        self.feature_columns = None
        self.target_column = None
        
    def process(self, data, task="classification", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "classification":
            return self.train_classifier(data, **kwargs)
        elif task == "regression":
            return self.train_regressor(data, **kwargs)
        elif task == "predict":
            return self.predict(data, **kwargs)
        else:
            return f"Supervised learning task '{task}' not supported."
    
    def train_classifier(self, data, target_column=None, feature_columns=None, 
                         algorithm="random_forest", test_size=0.2, random_state=42, **kwargs):
        """
        Train a classification model
        
        Args:
            data (pd.DataFrame): Input dataframe with features and target
            target_column (str): Name of the target column
            feature_columns (list): List of feature column names
            algorithm (str): Algorithm to use (random_forest, logistic, svm, decision_tree, xgboost)
            test_size (float): Test split ratio
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary with model performance metrics
        """
        if isinstance(data, pd.DataFrame):
            # Set target and feature columns
            self.target_column = target_column or data.columns[-1]
            if feature_columns:
                self.feature_columns = feature_columns
            else:
                self.feature_columns = [col for col in data.columns if col != self.target_column]
                
            # Prepare data
            X = data[self.feature_columns]
            y = data[self.target_column]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize model based on algorithm
            if algorithm == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    random_state=random_state
                )
            elif algorithm == "logistic":
                self.model = LogisticRegression(
                    random_state=random_state,
                    max_iter=kwargs.get('max_iter', 1000)
                )
            elif algorithm == "svm":
                self.model = SVC(
                    kernel=kwargs.get('kernel', 'rbf'),
                    probability=True,
                    random_state=random_state
                )
            elif algorithm == "decision_tree":
                self.model = DecisionTreeClassifier(
                    max_depth=kwargs.get('max_depth', None),
                    random_state=random_state
                )
            elif algorithm == "xgboost":
                self.model = xgb.XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=random_state
                )
            else:
                return f"Classification algorithm '{algorithm}' not supported."
                
            # Train model
            self.model.fit(X_train, y_train)
            self.model_type = "classifier"
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            return {
                "algorithm": algorithm,
                "model_type": "classifier",
                "accuracy": accuracy,
                "classification_report": report,
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_scores.mean(),
                "feature_importance": self._get_feature_importance() if hasattr(self.model, 'feature_importances_') else None
            }
        else:
            return "Data should be a pandas DataFrame."
    
    def train_regressor(self, data, target_column=None, feature_columns=None, 
                        algorithm="random_forest", test_size=0.2, random_state=42, **kwargs):
        """
        Train a regression model
        
        Args:
            data (pd.DataFrame): Input dataframe with features and target
            target_column (str): Name of the target column
            feature_columns (list): List of feature column names
            algorithm (str): Algorithm to use (random_forest, linear, svm, decision_tree, xgboost)
            test_size (float): Test split ratio
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary with model performance metrics
        """
        if isinstance(data, pd.DataFrame):
            # Set target and feature columns
            self.target_column = target_column or data.columns[-1]
            if feature_columns:
                self.feature_columns = feature_columns
            else:
                self.feature_columns = [col for col in data.columns if col != self.target_column]
                
            # Prepare data
            X = data[self.feature_columns]
            y = data[self.target_column]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize model based on algorithm
            if algorithm == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    random_state=random_state
                )
            elif algorithm == "linear":
                self.model = LinearRegression()
            elif algorithm == "svm":
                self.model = SVR(
                    kernel=kwargs.get('kernel', 'rbf')
                )
            elif algorithm == "decision_tree":
                self.model = DecisionTreeRegressor(
                    max_depth=kwargs.get('max_depth', None),
                    random_state=random_state
                )
            elif algorithm == "xgboost":
                self.model = xgb.XGBRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=random_state
                )
            else:
                return f"Regression algorithm '{algorithm}' not supported."
                
            # Train model
            self.model.fit(X_train, y_train)
            self.model_type = "regressor"
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores).mean()
            
            return {
                "algorithm": algorithm,
                "model_type": "regressor",
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "cv_rmse": cv_rmse,
                "feature_importance": self._get_feature_importance() if hasattr(self.model, 'feature_importances_') else None
            }
        else:
            return "Data should be a pandas DataFrame."
    
    def predict(self, data, **kwargs):
        """
        Make predictions using the trained model
        
        Args:
            data: Input data (pandas DataFrame or numpy array)
            
        Returns:
            Predictions
        """
        if self.model is None:
            return "No model has been trained yet."
            
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            # If dataframe has exact matching feature columns
            if all(col in data.columns for col in self.feature_columns):
                X = data[self.feature_columns]
            else:
                # Attempt to use all columns
                X = data
        elif isinstance(data, np.ndarray):
            X = data
        else:
            return "Data should be a pandas DataFrame or numpy array."
            
        # Make predictions
        if self.model_type == "classifier":
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
            
            result = {"predictions": predictions.tolist()}
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist()
            return result
        else:  # regressor
            predictions = self.model.predict(X)
            return {"predictions": predictions.tolist()}
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        if self.model is None:
            return "No model has been trained yet."
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Save metadata
            metadata = {
                "model_type": self.model_type,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column
            }
            metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.joblib"
            joblib.dump(metadata, metadata_path)
            
            return f"Model and metadata saved to {filepath} and {metadata_path}"
        except Exception as e:
            return f"Error saving model: {str(e)}"
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        try:
            # Load model
            self.model = joblib.load(filepath)
            
            # Load metadata
            metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.model_type = metadata.get("model_type")
                self.feature_columns = metadata.get("feature_columns")
                self.target_column = metadata.get("target_column")
                
            return "Model loaded successfully"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def _get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, feature_importance.tolist()))
        return None
