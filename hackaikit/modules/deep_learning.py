from hackaikit.core.base_module import BaseModule
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming input image is 32x32
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, input_dim),
            nn.Sigmoid()  # Assuming inputs are normalized between 0 and 1
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class SimpleGAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        
class TabularDataset(Dataset):
    def __init__(self, data, target_col=None, transform=None):
        """
        Dataset for tabular data
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            transform: Optional transforms to apply
        """
        self.transform = transform
        
        if target_col:
            self.X = data.drop(target_col, axis=1).values.astype(np.float32)
            self.y = data[target_col].values.astype(np.int64)
        else:
            self.X = data.values.astype(np.float32)
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
            
        if self.y is not None:
            return x, self.y[idx]
        return x

class DeepLearningModule(BaseModule):
    """
    Module for deep learning tasks using PyTorch.
    Supports various neural network architectures.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.model = None
        self.model_type = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.criterion = None
        self.input_dim = None
        self.output_dim = None
        
    def process(self, data, task="train", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "train":
            return self.train_model(data, **kwargs)
        elif task == "predict":
            return self.predict(data, **kwargs)
        elif task == "evaluate":
            return self.evaluate_model(data, **kwargs)
        else:
            return f"Deep learning task '{task}' not supported."
    
    def train_model(self, data, model_type="classifier", input_dim=None, output_dim=None, 
                    hidden_dim=128, batch_size=32, epochs=10, learning_rate=0.001, **kwargs):
        """
        Train a deep learning model
        
        Args:
            data: Input data (pandas DataFrame, numpy array, or torch Dataset)
            model_type (str): Type of model (classifier, cnn, autoencoder)
            input_dim (int): Input dimension
            output_dim (int): Output dimension (number of classes for classifier)
            hidden_dim (int): Hidden dimension
            batch_size (int): Batch size
            epochs (int): Number of epochs
            learning_rate (float): Learning rate
            
        Returns:
            dict: Dictionary with training results
        """
        # Set device
        if kwargs.get('use_gpu', True) and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            target_col = kwargs.get('target_col')
            dataset = TabularDataset(data, target_col=target_col)
        elif isinstance(data, np.ndarray):
            dataset = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, Dataset):
            dataset = data
        else:
            return "Data type not supported. Use pandas DataFrame, numpy array, or torch Dataset."
        
        # Create dataloader
        if isinstance(dataset, Dataset):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            # Handle tensor data
            dataloader = [(dataset, None)]
        
        # Initialize model based on type
        self.model_type = model_type
        
        # Determine input and output dimensions if not provided
        if input_dim is None:
            if hasattr(dataset, 'X'):
                input_dim = dataset.X.shape[1]
            else:
                input_dim = dataset.shape[1]
        self.input_dim = input_dim
        
        if output_dim is None and hasattr(dataset, 'y'):
            output_dim = len(np.unique(dataset.y))
        elif output_dim is None:
            output_dim = 1  # Default
        self.output_dim = output_dim
        
        if model_type == "classifier":
            self.model = SimpleClassifier(input_dim, hidden_dim, output_dim)
            self.criterion = nn.CrossEntropyLoss()
        elif model_type == "cnn":
            input_channels = kwargs.get('input_channels', 3)
            self.model = SimpleCNN(input_channels, output_dim)
            self.criterion = nn.CrossEntropyLoss()
        elif model_type == "autoencoder":
            self.model = SimpleAutoencoder(input_dim, hidden_dim)
            self.criterion = nn.MSELoss()
        else:
            return f"Model type '{model_type}' not supported."
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set optimizer
        optimizer_type = kwargs.get('optimizer', 'adam')
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data in dataloader:
                # Get data
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    if targets is not None:
                        targets = targets.to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                    targets = inputs  # For autoencoder
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if model_type == "autoencoder":
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for this epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        return {
            "model_type": model_type,
            "device": str(self.device),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "final_loss": losses[-1],
            "loss_history": losses
        }
    
    def predict(self, data, **kwargs):
        """
        Make predictions using the trained model
        
        Args:
            data: Input data (pandas DataFrame, numpy array, or torch tensor)
            
        Returns:
            Predictions
        """
        if self.model is None:
            return "No model has been trained yet."
            
        # Prepare input data
        if isinstance(data, pd.DataFrame):
            # If target column is in the data, remove it
            target_col = kwargs.get('target_col')
            if target_col and target_col in data.columns:
                X = data.drop(target_col, axis=1).values.astype(np.float32)
            else:
                X = data.values.astype(np.float32)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        elif isinstance(data, np.ndarray):
            X = torch.tensor(data, dtype=torch.float32).to(self.device)
        elif isinstance(data, torch.Tensor):
            X = data.to(self.device)
        else:
            return "Data type not supported. Use pandas DataFrame, numpy array, or torch tensor."
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            
            if self.model_type == "classifier":
                _, predicted = torch.max(outputs, 1)
                return {
                    "predictions": predicted.cpu().numpy().tolist(),
                    "probabilities": nn.Softmax(dim=1)(outputs).cpu().numpy().tolist()
                }
            elif self.model_type == "autoencoder":
                return {
                    "reconstructed": outputs.cpu().numpy().tolist(),
                    "encoded": self.model.encode(X).cpu().numpy().tolist()
                }
            else:
                return {
                    "predictions": outputs.cpu().numpy().tolist()
                }
    
    def evaluate_model(self, data, target_col=None, **kwargs):
        """
        Evaluate the model on test data
        
        Args:
            data (pd.DataFrame): Test data
            target_col (str): Name of the target column
            
        Returns:
            dict: Dictionary with evaluation results
        """
        if self.model is None:
            return "No model has been trained yet."
            
        # Prepare data
        if isinstance(data, pd.DataFrame) and target_col:
            X = data.drop(target_col, axis=1).values.astype(np.float32)
            y = data[target_col].values
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            if self.model_type == "classifier":
                y = torch.tensor(y, dtype=torch.long).to(self.device)
            else:
                y = torch.tensor(y, dtype=torch.float32).to(self.device)
        else:
            return "For evaluation, provide a pandas DataFrame with a target column."
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X)
            
            # Calculate loss
            loss = self.criterion(outputs, y).item()
            
            # For classification, calculate accuracy
            if self.model_type == "classifier":
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y).sum().item()
                accuracy = correct / y.size(0)
                
                # Calculate per-class accuracy
                class_correct = [0] * self.output_dim
                class_total = [0] * self.output_dim
                
                for i in range(y.size(0)):
                    label = y[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                
                class_accuracy = [
                    class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                    for i in range(self.output_dim)
                ]
                
                return {
                    "loss": loss,
                    "accuracy": accuracy,
                    "class_accuracy": class_accuracy,
                    "confusion_matrix": self._compute_confusion_matrix(y.cpu().numpy(), predicted.cpu().numpy())
                }
            
            # For regression or autoencoder, return MSE or other metrics
            else:
                return {
                    "loss": loss,
                    "mse": loss  # For MSE loss
                }
    
    def save_model(self, path):
        """Save model to disk"""
        if self.model is None:
            return "No model has been trained yet."
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state dict
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_type': self.model_type,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim
            }
            
            torch.save(model_state, path)
            return f"Model saved to {path}"
        except Exception as e:
            return f"Error saving model: {str(e)}"
    
    def load_model(self, path, hidden_dim=128):
        """Load model from disk"""
        try:
            # Load model state dict
            checkpoint = torch.load(path, map_location=self.device)
            
            # Get model parameters
            self.model_type = checkpoint.get('model_type', 'classifier')
            self.input_dim = checkpoint.get('input_dim', 10)
            self.output_dim = checkpoint.get('output_dim', 2)
            
            # Initialize model based on type
            if self.model_type == "classifier":
                self.model = SimpleClassifier(self.input_dim, hidden_dim, self.output_dim)
                self.criterion = nn.CrossEntropyLoss()
            elif self.model_type == "cnn":
                input_channels = 3  # Default
                self.model = SimpleCNN(input_channels, self.output_dim)
                self.criterion = nn.CrossEntropyLoss()
            elif self.model_type == "autoencoder":
                self.model = SimpleAutoencoder(self.input_dim, hidden_dim)
                self.criterion = nn.MSELoss()
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            return f"Model loaded from {path}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def _compute_confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix for classification"""
        n_classes = self.output_dim
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            confusion_matrix[y_true[i]][y_pred[i]] += 1
            
        return confusion_matrix.tolist()
