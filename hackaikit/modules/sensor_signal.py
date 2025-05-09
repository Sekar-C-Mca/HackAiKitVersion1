from hackaikit.core.base_module import BaseModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import os
import json
import pickle
from typing import Dict, List, Union, Optional, Any
import warnings

class SensorSignalModule(BaseModule):
    """
    Module for processing sensor and signal data for IoT applications and anomaly detection.
    Supports time series data preprocessing, feature extraction, anomaly detection, and signal processing.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.scaler = None
        self.model = None
        self.model_type = None
        self.pca = None
        self.n_components = None
        self.feature_columns = None
        self.timestamp_column = None
        
    def process(self, data, task="anomaly_detection", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "anomaly_detection":
            return self.detect_anomalies(data, **kwargs)
        elif task == "preprocess":
            return self.preprocess_data(data, **kwargs)
        elif task == "signal_processing":
            return self.process_signal(data, **kwargs)
        elif task == "feature_extraction":
            return self.extract_features(data, **kwargs)
        elif task == "visualize":
            return self.visualize_data(data, **kwargs)
        else:
            return f"Sensor signal task '{task}' not supported."
    
    def preprocess_data(self, data, timestamp_column=None, feature_columns=None, 
                        resample=None, fillna_method='interpolate', filter_type=None, **kwargs):
        """
        Preprocess sensor time series data
        
        Args:
            data (pd.DataFrame): Input dataframe with time series data
            timestamp_column (str): Name of the timestamp column
            feature_columns (list): List of feature column names
            resample (str): Pandas resample frequency (e.g., '1min', '1h')
            fillna_method (str): Method to fill NaN values (interpolate, ffill, bfill)
            filter_type (str): Filter type (lowpass, highpass, bandpass)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        if not isinstance(data, pd.DataFrame):
            return "Data should be a pandas DataFrame."
            
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Set timestamp and feature columns
        self.timestamp_column = timestamp_column or df.columns[0]
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            # Assume all numeric columns except timestamp are features
            self.feature_columns = [col for col in df.columns if col != self.timestamp_column 
                                   and pd.api.types.is_numeric_dtype(df[col])]
        
        # Convert timestamp column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_column]):
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
        
        # Set timestamp as index
        df = df.set_index(self.timestamp_column)
        
        # Resample data if specified
        if resample:
            agg_dict = {col: 'mean' for col in self.feature_columns}
            df = df.resample(resample).agg(agg_dict)
        
        # Fill NaN values
        if fillna_method == 'interpolate':
            df = df.interpolate(method='time')
        elif fillna_method == 'ffill':
            df = df.ffill()
        elif fillna_method == 'bfill':
            df = df.bfill()
        
        # Apply signal filtering if specified
        if filter_type:
            for col in self.feature_columns:
                df[col] = self._apply_filter(df[col].values, filter_type, **kwargs)
        
        # Handle any remaining NaN values
        df = df.dropna()
        
        return df
    
    def detect_anomalies(self, data, method="isolation_forest", timestamp_column=None, 
                         feature_columns=None, contamination=0.05, **kwargs):
        """
        Detect anomalies in sensor data
        
        Args:
            data (pd.DataFrame): Input dataframe with sensor data
            method (str): Anomaly detection method (isolation_forest, lof, dbscan)
            timestamp_column (str): Name of the timestamp column
            feature_columns (list): List of feature column names
            contamination (float): Expected proportion of anomalies
            
        Returns:
            dict: Dictionary with anomaly detection results
        """
        # Preprocess data
        df = self.preprocess_data(data, timestamp_column, feature_columns, **kwargs)
        
        if isinstance(df, str):  # Error message
            return df
            
        # Scale the data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[self.feature_columns])
        
        # Apply dimensionality reduction if there are many features
        if len(self.feature_columns) > 10:
            self.n_components = min(len(self.feature_columns), 10)
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)
        
        # Detect anomalies based on method
        if method == "isolation_forest":
            self.model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42),
                n_estimators=kwargs.get('n_estimators', 100)
            )
            self.model_type = "isolation_forest"
            
        elif method == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=contamination,
                novelty=False  # LOF in sklearn doesn't support predict on new data when novelty=False
            )
            self.model_type = "lof"
            
        elif method == "dbscan":
            self.model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
            self.model_type = "dbscan"
            
        else:
            return f"Anomaly detection method '{method}' not supported."
        
        # Train model and get predictions
        if method == "isolation_forest":
            self.model.fit(X)
            scores = self.model.decision_function(X)
            predictions = self.model.predict(X)
            # Convert predictions: +1 for inliers, -1 for outliers
            anomalies = np.where(predictions == -1, 1, 0)
            
        elif method == "lof":
            predictions = self.model.fit_predict(X)
            # LOF gives negative values for outliers
            anomalies = np.where(predictions == -1, 1, 0)
            # Get negative outlier factors (more negative means more anomalous)
            scores = -self.model.negative_outlier_factor_
            
        elif method == "dbscan":
            predictions = self.model.fit_predict(X)
            # DBSCAN marks outliers as -1
            anomalies = np.where(predictions == -1, 1, 0)
            # For DBSCAN, compute a score based on distance to nearest core point
            scores = np.zeros(len(X))
            
            if hasattr(self.model, 'components_'):
                for i in range(len(X)):
                    if predictions[i] == -1:  # For outliers
                        # Compute distance to nearest cluster
                        min_dist = np.min([np.min(np.linalg.norm(X[i] - self.model.components_, axis=1)), 
                                         np.inf])
                        scores[i] = -min_dist  # Negative distance (more negative is more anomalous)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['anomaly'] = anomalies
        results_df['score'] = scores
        
        # Get anomaly indices and timestamps
        anomaly_indices = np.where(anomalies == 1)[0]
        anomaly_timestamps = results_df.index[anomaly_indices].tolist()
        
        # Reset index to make the timestamp column available again
        results_df = results_df.reset_index()
        
        return {
            "method": method,
            "data_with_anomalies": results_df,
            "anomaly_indices": anomaly_indices.tolist(),
            "anomaly_timestamps": anomaly_timestamps,
            "num_anomalies": len(anomaly_indices),
            "anomaly_percentage": (len(anomaly_indices) / len(results_df)) * 100
        }
    
    def process_signal(self, data, signal_column=None, method="fft", 
                      filter_type=None, window_size=None, **kwargs):
        """
        Process signal data with various signal processing techniques
        
        Args:
            data (pd.DataFrame or np.ndarray): Input data with signal
            signal_column (str): Name of the signal column if data is DataFrame
            method (str): Signal processing method (fft, stft, filter)
            filter_type (str): Filter type if method is filter (lowpass, highpass, bandpass)
            window_size (int): Window size for methods that need it
            
        Returns:
            dict: Dictionary with signal processing results
        """
        # Extract signal from data
        if isinstance(data, pd.DataFrame) and signal_column:
            signal_data = data[signal_column].values
            sampling_freq = kwargs.get('sampling_freq', 1.0)
            
            # Try to infer sampling frequency from timestamp if available
            timestamp_col = kwargs.get('timestamp_column')
            if timestamp_col and timestamp_col in data.columns:
                times = pd.to_datetime(data[timestamp_col])
                if len(times) > 1:
                    # Calculate average time difference in seconds
                    avg_diff = (times.iloc[-1] - times.iloc[0]).total_seconds() / (len(times) - 1)
                    sampling_freq = 1.0 / avg_diff
                    
        elif isinstance(data, np.ndarray):
            signal_data = data
            sampling_freq = kwargs.get('sampling_freq', 1.0)
        else:
            return "Invalid data format. Provide DataFrame with signal_column or numpy array."
        
        # Apply signal processing based on method
        if method == "fft":
            return self._compute_fft(signal_data, sampling_freq)
            
        elif method == "stft":
            return self._compute_stft(signal_data, sampling_freq, window_size)
            
        elif method == "filter":
            if not filter_type:
                return "Filter type must be specified for filter method."
                
            filtered_signal = self._apply_filter(signal_data, filter_type, **kwargs)
            
            return {
                "method": "filter",
                "filter_type": filter_type,
                "original_signal": signal_data.tolist(),
                "filtered_signal": filtered_signal.tolist()
            }
            
        elif method == "spectrogram":
            frequencies, times, spectrogram = signal.spectrogram(
                signal_data, 
                fs=sampling_freq,
                nperseg=window_size or 256
            )
            
            # Convert to dB for better visualization
            spectrogram_db = 10 * np.log10(spectrogram)
            
            return {
                "method": "spectrogram",
                "frequencies": frequencies.tolist(),
                "times": times.tolist(),
                "spectrogram": spectrogram_db.tolist()
            }
            
        else:
            return f"Signal processing method '{method}' not supported."
    
    def extract_features(self, data, timestamp_column=None, feature_columns=None, 
                        window_size=None, overlap=0.5, **kwargs):
        """
        Extract features from sensor data
        
        Args:
            data (pd.DataFrame): Input dataframe with sensor data
            timestamp_column (str): Name of the timestamp column
            feature_columns (list): List of feature column names
            window_size (int): Window size for feature extraction
            overlap (float): Overlap between windows (0 to 1)
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        # Preprocess data
        df = self.preprocess_data(data, timestamp_column, feature_columns, **kwargs)
        
        if isinstance(df, str):  # Error message
            return df
            
        # If no window size is provided, use whole dataset
        if window_size is None:
            # Extract features from the entire dataset
            features = {}
            
            for col in self.feature_columns:
                col_data = df[col].values
                
                # Time domain features
                features[f"{col}_mean"] = np.mean(col_data)
                features[f"{col}_std"] = np.std(col_data)
                features[f"{col}_min"] = np.min(col_data)
                features[f"{col}_max"] = np.max(col_data)
                features[f"{col}_range"] = np.max(col_data) - np.min(col_data)
                features[f"{col}_median"] = np.median(col_data)
                features[f"{col}_skew"] = 0 if len(col_data) < 3 else pd.Series(col_data).skew()
                features[f"{col}_kurtosis"] = 0 if len(col_data) < 4 else pd.Series(col_data).kurtosis()
                features[f"{col}_rms"] = np.sqrt(np.mean(np.square(col_data)))
                
                # Frequency domain features if enough data points
                if len(col_data) >= 10:
                    fft_result = np.abs(fft(col_data))
                    freq = fftfreq(len(col_data))
                    
                    features[f"{col}_fft_max"] = np.max(fft_result)
                    features[f"{col}_fft_mean"] = np.mean(fft_result)
                    features[f"{col}_fft_std"] = np.std(fft_result)
                    
                    # Dominant frequency
                    dom_freq_idx = np.argmax(fft_result[1:len(fft_result)//2]) + 1
                    features[f"{col}_dominant_freq"] = freq[dom_freq_idx]
                    features[f"{col}_dominant_freq_amplitude"] = fft_result[dom_freq_idx]
            
            # Return as a single-row DataFrame
            return pd.DataFrame([features])
        
        else:
            # Extract features from sliding windows
            all_features = []
            
            # Calculate step size
            step_size = int(window_size * (1 - overlap))
            
            # Iterate through the data in windows
            for i in range(0, len(df) - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                window_start_time = window.index[0]
                window_end_time = window.index[-1]
                
                # Extract features from this window
                window_features = {
                    'window_start': window_start_time,
                    'window_end': window_end_time
                }
                
                for col in self.feature_columns:
                    col_data = window[col].values
                    
                    # Time domain features
                    window_features[f"{col}_mean"] = np.mean(col_data)
                    window_features[f"{col}_std"] = np.std(col_data)
                    window_features[f"{col}_min"] = np.min(col_data)
                    window_features[f"{col}_max"] = np.max(col_data)
                    window_features[f"{col}_range"] = np.max(col_data) - np.min(col_data)
                    window_features[f"{col}_median"] = np.median(col_data)
                    window_features[f"{col}_skew"] = 0 if len(col_data) < 3 else pd.Series(col_data).skew()
                    window_features[f"{col}_kurtosis"] = 0 if len(col_data) < 4 else pd.Series(col_data).kurtosis()
                    window_features[f"{col}_rms"] = np.sqrt(np.mean(np.square(col_data)))
                    
                    # Frequency domain features
                    fft_result = np.abs(fft(col_data))
                    freq = fftfreq(len(col_data))
                    
                    window_features[f"{col}_fft_max"] = np.max(fft_result)
                    window_features[f"{col}_fft_mean"] = np.mean(fft_result)
                    window_features[f"{col}_fft_std"] = np.std(fft_result)
                    
                    # Dominant frequency
                    dom_freq_idx = np.argmax(fft_result[1:len(fft_result)//2]) + 1
                    window_features[f"{col}_dominant_freq"] = freq[dom_freq_idx]
                    window_features[f"{col}_dominant_freq_amplitude"] = fft_result[dom_freq_idx]
                
                all_features.append(window_features)
            
            # Return DataFrame with features from all windows
            features_df = pd.DataFrame(all_features)
            
            # Set window start time as index
            if 'window_start' in features_df.columns:
                features_df = features_df.set_index('window_start')
            
            return features_df
    
    def visualize_data(self, data, plot_type="time_series", timestamp_column=None, 
                      feature_columns=None, anomalies=None, save_path=None, **kwargs):
        """
        Visualize sensor data in different ways
        
        Args:
            data (pd.DataFrame): Input dataframe with sensor data
            plot_type (str): Type of plot (time_series, heatmap, fft, scatter)
            timestamp_column (str): Name of the timestamp column
            feature_columns (list): List of feature column names
            anomalies (pd.Series or list): Boolean indicators or indices of anomalies
            save_path (str): Path to save the visualization
            
        Returns:
            str: Path to saved visualization or message
        """
        if not isinstance(data, pd.DataFrame):
            return "Data should be a pandas DataFrame."
            
        # Set timestamp and feature columns
        timestamp_column = timestamp_column or data.columns[0]
        if feature_columns:
            feature_cols = feature_columns
        else:
            # Assume all numeric columns except timestamp are features
            feature_cols = [col for col in data.columns if col != timestamp_column 
                           and pd.api.types.is_numeric_dtype(data[col])]
        
        # Make a copy and ensure timestamp is in datetime format
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Set figure size
        figsize = kwargs.get('figsize', (12, 8))
        plt.figure(figsize=figsize)
        
        # Create visualization based on type
        if plot_type == "time_series":
            # Set timestamp as index if it's not already
            if df.index.name != timestamp_column:
                df = df.set_index(timestamp_column)
            
            # Plot each feature
            for col in feature_cols:
                plt.plot(df.index, df[col], label=col)
            
            # Add anomalies if provided
            if anomalies is not None:
                if isinstance(anomalies, pd.Series):
                    anomaly_indices = anomalies[anomalies == 1].index
                    
                    for col in feature_cols:
                        plt.scatter(anomaly_indices, df.loc[anomaly_indices, col], 
                                   color='red', label=f'{col} Anomalies' if col == feature_cols[0] else "")
                        
                elif isinstance(anomalies, list):
                    # Assuming list of indices
                    for col in feature_cols:
                        plt.scatter(df.index[anomalies], df.iloc[anomalies][col], 
                                   color='red', label=f'{col} Anomalies' if col == feature_cols[0] else "")
            
            plt.title('Sensor Data Time Series', fontsize=14)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif plot_type == "heatmap":
            import seaborn as sns
            
            # Set timestamp as index if it's not already
            if df.index.name != timestamp_column:
                df = df.set_index(timestamp_column)
            
            # Create correlation matrix
            correlation = df[feature_cols].corr()
            
            # Plot heatmap
            sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Feature Correlation Heatmap', fontsize=14)
            
        elif plot_type == "fft":
            # Only use the first feature if multiple are provided
            feature = feature_cols[0]
            
            # Compute FFT
            signal_data = df[feature].values
            fft_result = np.abs(fft(signal_data))
            freq = fftfreq(len(signal_data))
            
            # Plot only positive frequencies
            positive_freq_mask = freq > 0
            plt.plot(freq[positive_freq_mask], fft_result[positive_freq_mask])
            
            plt.title(f'FFT of {feature}', fontsize=14)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.grid(True, alpha=0.3)
            
        elif plot_type == "scatter":
            if len(feature_cols) < 2:
                return "At least two feature columns are required for scatter plot."
                
            x_feature = kwargs.get('x_feature', feature_cols[0])
            y_feature = kwargs.get('y_feature', feature_cols[1])
            
            plt.scatter(df[x_feature], df[y_feature], alpha=0.7)
            
            # Add anomalies if provided
            if anomalies is not None:
                if isinstance(anomalies, pd.Series):
                    anomaly_indices = anomalies[anomalies == 1].index
                    plt.scatter(df.loc[anomaly_indices, x_feature], 
                               df.loc[anomaly_indices, y_feature], 
                               color='red', label='Anomalies')
                elif isinstance(anomalies, list):
                    plt.scatter(df.iloc[anomalies][x_feature], 
                               df.iloc[anomalies][y_feature], 
                               color='red', label='Anomalies')
            
            plt.title(f'Scatter Plot: {x_feature} vs {y_feature}', fontsize=14)
            plt.xlabel(x_feature, fontsize=12)
            plt.ylabel(y_feature, fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            return f"Plot type '{plot_type}' not supported."
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return f"Visualization saved to {save_path}"
            except Exception as e:
                plt.close()
                return f"Error saving visualization: {str(e)}"
        
        plt.show()
        return "Visualization displayed"
    
    def save_model(self, filepath):
        """Save the anomaly detection model to a file"""
        if self.model is None:
            return "No model to save. Run anomaly detection first."
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a model state dictionary
            model_state = {
                'model_type': self.model_type,
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'timestamp_column': self.timestamp_column
            }
            
            # Add PCA if used
            if self.pca is not None:
                model_state['pca'] = self.pca
                model_state['n_components'] = self.n_components
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            return f"Model saved to {filepath}"
            
        except Exception as e:
            return f"Error saving model: {str(e)}"
    
    def load_model(self, filepath):
        """Load an anomaly detection model from a file"""
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            self.model_type = model_state['model_type']
            self.model = model_state['model']
            self.scaler = model_state['scaler']
            self.feature_columns = model_state['feature_columns']
            self.timestamp_column = model_state['timestamp_column']
            
            # Load PCA if present
            if 'pca' in model_state:
                self.pca = model_state['pca']
                self.n_components = model_state['n_components']
            
            return f"Model loaded from {filepath}"
            
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def _compute_fft(self, signal_data, sampling_freq):
        """Compute FFT of signal data"""
        # Compute FFT
        fft_result = fft(signal_data)
        
        # Compute frequencies
        n = len(signal_data)
        freq = fftfreq(n, 1/sampling_freq)
        
        # Get only positive frequencies
        positive_freq_mask = freq >= 0
        positive_freq = freq[positive_freq_mask]
        positive_fft = np.abs(fft_result[positive_freq_mask])
        
        # Find dominant frequency
        max_idx = np.argmax(positive_fft)
        dominant_freq = positive_freq[max_idx]
        dominant_amplitude = positive_fft[max_idx]
        
        return {
            "method": "fft",
            "frequencies": positive_freq.tolist(),
            "amplitudes": positive_fft.tolist(),
            "dominant_frequency": dominant_freq,
            "dominant_amplitude": dominant_amplitude
        }
    
    def _compute_stft(self, signal_data, sampling_freq, window_size=None):
        """Compute Short-Time Fourier Transform of signal data"""
        if window_size is None:
            window_size = min(256, len(signal_data) // 8)
            
        # Compute STFT
        f, t, Zxx = signal.stft(signal_data, fs=sampling_freq, nperseg=window_size)
        
        # Get magnitude
        magnitude = np.abs(Zxx)
        
        return {
            "method": "stft",
            "frequencies": f.tolist(),
            "times": t.tolist(),
            "magnitude": magnitude.tolist()
        }
    
    def _apply_filter(self, signal_data, filter_type, **kwargs):
        """Apply filter to signal data"""
        # Get filter parameters
        order = kwargs.get('filter_order', 4)
        nyquist = 0.5 * kwargs.get('sampling_freq', 1.0)
        
        if filter_type == "lowpass":
            cutoff = kwargs.get('cutoff_freq', 0.1) / nyquist
            b, a = signal.butter(order, cutoff, btype='lowpass')
            
        elif filter_type == "highpass":
            cutoff = kwargs.get('cutoff_freq', 0.1) / nyquist
            b, a = signal.butter(order, cutoff, btype='highpass')
            
        elif filter_type == "bandpass":
            low_cutoff = kwargs.get('low_cutoff_freq', 0.1) / nyquist
            high_cutoff = kwargs.get('high_cutoff_freq', 0.4) / nyquist
            b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass')
            
        else:
            raise ValueError(f"Filter type '{filter_type}' not supported.")
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
