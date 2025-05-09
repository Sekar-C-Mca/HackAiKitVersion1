from hackaikit.core.base_module import BaseModule
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTMForecaster(nn.Module):
    """Basic LSTM forecaster for time series"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesModule(BaseModule):
    """
    Module for time series forecasting using various methods.
    Supports ARIMA, Prophet, Exponential Smoothing, and LSTM.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.model = None
        self.model_type = None
        self.scaler = None
        self.history = None
        self.freq = None
        self.forecast_steps = None
        self.time_column = None
        self.value_column = None
        self.exog_columns = None
        self.lstm_model = None
        self.sequence_length = None
        
    def process(self, data, task="forecast", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "forecast":
            return self.create_forecast(data, **kwargs)
        elif task == "train":
            return self.train_model(data, **kwargs)
        elif task == "evaluate":
            return self.evaluate_model(data, **kwargs)
        else:
            return f"Time series task '{task}' not supported."
    
    def train_model(self, data, model_type="arima", time_column=None, value_column=None, 
                    exog_columns=None, **kwargs):
        """
        Train a time series model
        
        Args:
            data (pd.DataFrame): Input dataframe with time and value columns
            model_type (str): Type of model (arima, sarima, prophet, exponential, lstm)
            time_column (str): Name of the time column
            value_column (str): Name of the value column
            exog_columns (list): List of exogenous variable columns
            
        Returns:
            dict: Dictionary with training results
        """
        if not isinstance(data, pd.DataFrame):
            return "Data should be a pandas DataFrame."
            
        # Set column names
        self.time_column = time_column or data.columns[0]
        self.value_column = value_column or data.columns[1]
        self.exog_columns = exog_columns
        
        # Prepare data
        data = self._prepare_data(data)
        
        # Train model based on type
        if model_type == "arima":
            return self._train_arima(data, **kwargs)
        elif model_type == "sarima":
            return self._train_sarima(data, **kwargs)
        elif model_type == "prophet":
            return self._train_prophet(data, **kwargs)
        elif model_type == "exponential":
            return self._train_exponential(data, **kwargs)
        elif model_type == "lstm":
            return self._train_lstm(data, **kwargs)
        else:
            return f"Time series model type '{model_type}' not supported."
    
    def create_forecast(self, data=None, steps=10, include_history=True, **kwargs):
        """
        Create a forecast using the trained model
        
        Args:
            data (pd.DataFrame): Optional new data
            steps (int): Number of steps to forecast
            include_history (bool): Whether to include history in the result
            
        Returns:
            dict: Dictionary with forecast results
        """
        if self.model is None:
            return "No model has been trained yet."
            
        self.forecast_steps = steps
        
        # If new data is provided, retrain the model
        if data is not None:
            self.train_model(data, model_type=self.model_type, **kwargs)
            
        # Create forecast based on model type
        if self.model_type == "arima":
            return self._forecast_arima(steps, include_history)
        elif self.model_type == "sarima":
            return self._forecast_sarima(steps, include_history)
        elif self.model_type == "prophet":
            return self._forecast_prophet(steps, include_history)
        elif self.model_type == "exponential":
            return self._forecast_exponential(steps, include_history)
        elif self.model_type == "lstm":
            return self._forecast_lstm(steps, include_history)
        else:
            return f"Forecasting for model type '{self.model_type}' not implemented."
    
    def evaluate_model(self, test_data, metric="rmse", **kwargs):
        """
        Evaluate the model on test data
        
        Args:
            test_data (pd.DataFrame): Test data
            metric (str): Evaluation metric (rmse, mae, r2)
            
        Returns:
            dict: Dictionary with evaluation results
        """
        if self.model is None:
            return "No model has been trained yet."
            
        if not isinstance(test_data, pd.DataFrame):
            return "Test data should be a pandas DataFrame."
            
        # Prepare test data
        test_data = self._prepare_data(test_data)
        
        # Get true values
        y_true = test_data[self.value_column].values
        
        # Generate predictions
        if self.model_type == "arima" or self.model_type == "sarima":
            # For ARIMA models, we predict each point based on history
            y_pred = []
            history = self.history.copy()
            
            for t in range(len(test_data)):
                model = ARIMA(history, order=self.order) if self.model_type == "arima" else \
                        SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order)
                model_fit = model.fit()
                output = model_fit.forecast(1)
                yhat = output[0]
                y_pred.append(yhat)
                history = pd.concat([history, test_data.iloc[t:t+1]])
        
        elif self.model_type == "prophet":
            # For Prophet, we create a future dataframe and predict
            future = pd.DataFrame({
                'ds': test_data.index,
                'y': test_data[self.value_column]
            })
            forecast = self.model.predict(future)
            y_pred = forecast['yhat'].values
        
        elif self.model_type == "exponential":
            # For Exponential Smoothing, we forecast for the test period
            y_pred = self.model.forecast(len(test_data)).values
        
        elif self.model_type == "lstm":
            # For LSTM, we need to create sequences
            X_test = self._create_sequences(test_data[self.value_column].values, self.sequence_length)
            
            # Scale inputs
            X_test_scaled = np.array([self.scaler.transform(x.reshape(-1, 1)).flatten() for x in X_test])
            
            # Convert to torch tensor
            X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(2)
            
            # Make predictions
            self.lstm_model.eval()
            with torch.no_grad():
                y_pred = self.lstm_model(X_test_tensor).numpy().flatten()
                
            # Inverse transform predictions
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Align predictions with true values
            y_true = y_true[self.sequence_length:]
        
        else:
            return f"Evaluation for model type '{self.model_type}' not implemented."
        
        # Calculate metrics
        results = {}
        if len(y_true) == len(y_pred):
            if metric == "rmse" or metric == "all":
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                results["rmse"] = rmse
                
            if metric == "mae" or metric == "all":
                mae = mean_absolute_error(y_true, y_pred)
                results["mae"] = mae
                
            if metric == "r2" or metric == "all":
                r2 = r2_score(y_true, y_pred)
                results["r2"] = r2
        else:
            results["error"] = "Prediction length doesn't match true values"
        
        return {
            "model_type": self.model_type,
            "true_values": y_true.tolist(),
            "predictions": y_pred.tolist(),
            "metrics": results
        }
    
    def visualize_forecast(self, steps=None, save_path=None, figsize=(12, 6), **kwargs):
        """
        Visualize the forecast with historical data
        
        Args:
            steps (int): Number of steps to forecast
            save_path (str): Path to save the visualization
            figsize (tuple): Figure size
            
        Returns:
            Path to the saved visualization or None
        """
        if self.model is None:
            return "No model has been trained yet."
            
        # Create forecast
        if steps is not None:
            self.forecast_steps = steps
            
        forecast_result = self.create_forecast(steps=self.forecast_steps, include_history=True)
        
        if isinstance(forecast_result, str):  # Error message
            return forecast_result
            
        # Extract data for plotting
        forecast_df = forecast_result.get("forecast")
        
        if forecast_df is None:
            return "No forecast data available."
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(forecast_df.index[:-self.forecast_steps], 
                 forecast_df[self.value_column][:-self.forecast_steps], 
                 'b-', label='Historical Data')
        
        # Plot forecast
        plt.plot(forecast_df.index[-self.forecast_steps:], 
                 forecast_df['forecast'][-self.forecast_steps:], 
                 'r--', label='Forecast')
        
        # If we have confidence intervals in the forecast
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            plt.fill_between(forecast_df.index[-self.forecast_steps:],
                             forecast_df['lower_bound'][-self.forecast_steps:],
                             forecast_df['upper_bound'][-self.forecast_steps:],
                             color='r', alpha=0.2, label='Confidence Interval')
        
        plt.title(f"{self.model_type.upper()} Forecast", fontsize=15)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(self.value_column, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if path is provided
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
        
        # If not saving, show the plot
        plt.show()
        return "Visualization displayed"
    
    def _prepare_data(self, data):
        """Prepare data for time series analysis"""
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # Set time column as index
        df = df.set_index(self.time_column)
        
        # Sort by index
        df = df.sort_index()
        
        # Infer frequency if not set
        if self.freq is None:
            self.freq = pd.infer_freq(df.index)
        
        # Handle missing values
        df = df.interpolate(method='time')
        
        return df
    
    def _train_arima(self, data, order=(1, 1, 1), **kwargs):
        """Train an ARIMA model"""
        try:
            # Save data for future use
            self.history = data.copy()
            self.order = order
            
            # Create and fit model
            model = ARIMA(data[self.value_column], order=order)
            self.model = model.fit()
            self.model_type = "arima"
            
            # Get model summary
            summary = self.model.summary()
            
            return {
                "model_type": "arima",
                "order": order,
                "aic": self.model.aic,
                "bic": self.model.bic
            }
        
        except Exception as e:
            return f"Error training ARIMA model: {str(e)}"
    
    def _train_sarima(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), **kwargs):
        """Train a SARIMA model"""
        try:
            # Save data for future use
            self.history = data.copy()
            self.order = order
            self.seasonal_order = seasonal_order
            
            # Create and fit model
            model = SARIMAX(data[self.value_column], order=order, seasonal_order=seasonal_order)
            self.model = model.fit(disp=False)
            self.model_type = "sarima"
            
            # Get model summary
            summary = self.model.summary()
            
            return {
                "model_type": "sarima",
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": self.model.aic,
                "bic": self.model.bic
            }
        
        except Exception as e:
            return f"Error training SARIMA model: {str(e)}"
    
    def _train_prophet(self, data, yearly_seasonality=True, weekly_seasonality=True, 
                       daily_seasonality=False, **kwargs):
        """Train a Prophet model"""
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data[self.value_column]
            })
            
            # Save data for future use
            self.history = df.copy()
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                **kwargs
            )
            
            # Add additional seasonality components if provided
            for seasonality in kwargs.get('seasonalities', []):
                model.add_seasonality(**seasonality)
            
            # Add country holidays if provided
            if 'country_holidays' in kwargs:
                model.add_country_holidays(country_name=kwargs['country_holidays'])
            
            # Fit model
            model.fit(df)
            self.model = model
            self.model_type = "prophet"
            
            return {
                "model_type": "prophet",
                "parameters": {
                    "yearly_seasonality": yearly_seasonality,
                    "weekly_seasonality": weekly_seasonality,
                    "daily_seasonality": daily_seasonality
                }
            }
        
        except Exception as e:
            return f"Error training Prophet model: {str(e)}"
    
    def _train_exponential(self, data, trend='add', seasonal='add', seasonal_periods=None, **kwargs):
        """Train an Exponential Smoothing model"""
        try:
            # Save data for future use
            self.history = data.copy()
            
            # Infer seasonal periods if not provided
            if seasonal_periods is None and self.freq:
                if self.freq.startswith('M'):  # Monthly data
                    seasonal_periods = 12
                elif self.freq.startswith('Q'):  # Quarterly data
                    seasonal_periods = 4
                elif self.freq.startswith('D'):  # Daily data
                    seasonal_periods = 7
                elif self.freq.startswith('B'):  # Business day data
                    seasonal_periods = 5
                elif self.freq.startswith('H'):  # Hourly data
                    seasonal_periods = 24
            
            # Create and fit model
            model = ExponentialSmoothing(
                data[self.value_column],
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                **kwargs
            )
            
            self.model = model.fit()
            self.model_type = "exponential"
            
            return {
                "model_type": "exponential",
                "parameters": {
                    "trend": trend,
                    "seasonal": seasonal,
                    "seasonal_periods": seasonal_periods
                },
                "aic": self.model.aic if hasattr(self.model, 'aic') else None
            }
        
        except Exception as e:
            return f"Error training Exponential Smoothing model: {str(e)}"
    
    def _train_lstm(self, data, sequence_length=10, hidden_size=50, num_layers=1, 
                    batch_size=16, num_epochs=100, learning_rate=0.001, **kwargs):
        """Train an LSTM model"""
        try:
            # Save data for future use
            self.history = data.copy()
            self.sequence_length = sequence_length
            
            # Get values as numpy array
            values = data[self.value_column].values
            
            # Scale data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            values_scaled = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
            
            # Create sequences for LSTM
            X, y = self._create_sequences(values_scaled, sequence_length, return_y=True)
            
            # Convert to torch tensors
            X_tensor = torch.FloatTensor(X).unsqueeze(2)  # [batch, seq_len, input_size]
            y_tensor = torch.FloatTensor(y).unsqueeze(1)  # [batch, output_size]
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Create model
            input_size = 1  # Single feature (time series value)
            output_size = 1  # Single prediction
            
            self.lstm_model = LSTMForecaster(input_size, hidden_size, num_layers, output_size)
            self.model_type = "lstm"
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
            
            # Training loop
            self.lstm_model.train()
            losses = []
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for X_batch, y_batch in dataloader:
                    # Forward pass
                    outputs = self.lstm_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Record average loss for this epoch
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
            # Save the model object for forecasting
            self.model = self.lstm_model
            
            return {
                "model_type": "lstm",
                "parameters": {
                    "sequence_length": sequence_length,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_epochs": num_epochs
                },
                "training_losses": losses
            }
        
        except Exception as e:
            return f"Error training LSTM model: {str(e)}"
    
    def _forecast_arima(self, steps, include_history):
        """Create forecasts for ARIMA"""
        try:
            # Make forecast
            forecast = self.model.forecast(steps=steps)
            
            # Create forecast dataframe
            if include_history:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with history and forecast
                forecast_df = pd.DataFrame(index=self.history.index.union(forecast_index))
                forecast_df[self.value_column] = self.history[self.value_column]
                forecast_df['forecast'] = np.nan
                forecast_df.loc[forecast_index, 'forecast'] = forecast
                
                # Add confidence intervals if available
                if hasattr(self.model, 'get_forecast'):
                    forecast_obj = self.model.get_forecast(steps=steps)
                    conf_int = forecast_obj.conf_int()
                    forecast_df.loc[forecast_index, 'lower_bound'] = conf_int.iloc[:, 0].values
                    forecast_df.loc[forecast_index, 'upper_bound'] = conf_int.iloc[:, 1].values
            else:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with just the forecast
                forecast_df = pd.DataFrame(index=forecast_index)
                forecast_df['forecast'] = forecast
                
                # Add confidence intervals if available
                if hasattr(self.model, 'get_forecast'):
                    forecast_obj = self.model.get_forecast(steps=steps)
                    conf_int = forecast_obj.conf_int()
                    forecast_df['lower_bound'] = conf_int.iloc[:, 0].values
                    forecast_df['upper_bound'] = conf_int.iloc[:, 1].values
            
            return {
                "model_type": "arima",
                "forecast": forecast_df
            }
        
        except Exception as e:
            return f"Error forecasting with ARIMA model: {str(e)}"
    
    def _forecast_sarima(self, steps, include_history):
        """Create forecasts for SARIMA"""
        try:
            # Make forecast
            forecast = self.model.forecast(steps=steps)
            
            # Create forecast dataframe
            if include_history:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with history and forecast
                forecast_df = pd.DataFrame(index=self.history.index.union(forecast_index))
                forecast_df[self.value_column] = self.history[self.value_column]
                forecast_df['forecast'] = np.nan
                forecast_df.loc[forecast_index, 'forecast'] = forecast
                
                # Add confidence intervals if available
                if hasattr(self.model, 'get_forecast'):
                    forecast_obj = self.model.get_forecast(steps=steps)
                    conf_int = forecast_obj.conf_int()
                    forecast_df.loc[forecast_index, 'lower_bound'] = conf_int.iloc[:, 0].values
                    forecast_df.loc[forecast_index, 'upper_bound'] = conf_int.iloc[:, 1].values
            else:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with just the forecast
                forecast_df = pd.DataFrame(index=forecast_index)
                forecast_df['forecast'] = forecast
                
                # Add confidence intervals if available
                if hasattr(self.model, 'get_forecast'):
                    forecast_obj = self.model.get_forecast(steps=steps)
                    conf_int = forecast_obj.conf_int()
                    forecast_df['lower_bound'] = conf_int.iloc[:, 0].values
                    forecast_df['upper_bound'] = conf_int.iloc[:, 1].values
            
            return {
                "model_type": "sarima",
                "forecast": forecast_df
            }
        
        except Exception as e:
            return f"Error forecasting with SARIMA model: {str(e)}"
    
    def _forecast_prophet(self, steps, include_history):
        """Create forecasts for Prophet"""
        try:
            # Create future dataframe
            last_date = self.history['ds'].iloc[-1]
            future = pd.DataFrame({
                'ds': pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
            })
            
            # Make forecast
            forecast = self.model.predict(future)
            
            # Create forecast dataframe
            if include_history:
                # Get historical predictions
                history_forecast = self.model.predict(self.history)
                
                # Combine history and forecast
                forecast_df = pd.DataFrame(index=pd.to_datetime(history_forecast['ds']).append(pd.to_datetime(forecast['ds'])))
                
                # Add historical values
                forecast_df[self.value_column] = np.nan
                forecast_df.loc[pd.to_datetime(self.history['ds']), self.value_column] = self.history['y'].values
                
                # Add historical fitted values and future forecast
                forecast_df['forecast'] = np.concatenate([history_forecast['yhat'].values, forecast['yhat'].values])
                
                # Add confidence intervals
                forecast_df['lower_bound'] = np.concatenate([history_forecast['yhat_lower'].values, forecast['yhat_lower'].values])
                forecast_df['upper_bound'] = np.concatenate([history_forecast['yhat_upper'].values, forecast['yhat_upper'].values])
            else:
                # Create dataframe with just the forecast
                forecast_df = pd.DataFrame(index=pd.to_datetime(forecast['ds']))
                forecast_df['forecast'] = forecast['yhat'].values
                forecast_df['lower_bound'] = forecast['yhat_lower'].values
                forecast_df['upper_bound'] = forecast['yhat_upper'].values
            
            return {
                "model_type": "prophet",
                "forecast": forecast_df,
                "components": {
                    "trend": forecast['trend'].values.tolist(),
                    "yearly": forecast['yearly'].values.tolist() if 'yearly' in forecast.columns else None,
                    "weekly": forecast['weekly'].values.tolist() if 'weekly' in forecast.columns else None,
                    "daily": forecast['daily'].values.tolist() if 'daily' in forecast.columns else None
                }
            }
        
        except Exception as e:
            return f"Error forecasting with Prophet model: {str(e)}"
    
    def _forecast_exponential(self, steps, include_history):
        """Create forecasts for Exponential Smoothing"""
        try:
            # Make forecast
            forecast = self.model.forecast(steps=steps)
            
            # Create forecast dataframe
            if include_history:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with history and forecast
                forecast_df = pd.DataFrame(index=self.history.index.union(forecast_index))
                forecast_df[self.value_column] = self.history[self.value_column]
                forecast_df['forecast'] = np.nan
                forecast_df.loc[forecast_index, 'forecast'] = forecast
                
                # Add fitted values for history
                fitted_values = self.model.fittedvalues
                forecast_df.loc[self.history.index, 'fitted'] = fitted_values
            else:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with just the forecast
                forecast_df = pd.DataFrame(index=forecast_index)
                forecast_df['forecast'] = forecast
            
            return {
                "model_type": "exponential",
                "forecast": forecast_df
            }
        
        except Exception as e:
            return f"Error forecasting with Exponential Smoothing model: {str(e)}"
    
    def _forecast_lstm(self, steps, include_history):
        """Create forecasts for LSTM"""
        try:
            # Get the last sequence from history
            last_sequence = self.history[self.value_column].values[-self.sequence_length:]
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
            
            # Initialize forecast array
            forecast_values = []
            current_sequence = last_sequence_scaled.copy()
            
            # Generate predictions iteratively
            self.lstm_model.eval()
            with torch.no_grad():
                for _ in range(steps):
                    # Prepare input tensor
                    current_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(2)
                    
                    # Get prediction for next step
                    next_value = self.lstm_model(current_tensor).item()
                    forecast_values.append(next_value)
                    
                    # Update sequence
                    current_sequence = np.append(current_sequence[1:], next_value)
            
            # Inverse transform predictions
            forecast_values = self.scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()
            
            # Create forecast dataframe
            if include_history:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with history and forecast
                forecast_df = pd.DataFrame(index=self.history.index.union(forecast_index))
                forecast_df[self.value_column] = self.history[self.value_column]
                forecast_df['forecast'] = np.nan
                forecast_df.loc[forecast_index, 'forecast'] = forecast_values
            else:
                # Create date range for forecast
                last_date = self.history.index[-1]
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
                
                # Create dataframe with just the forecast
                forecast_df = pd.DataFrame(index=forecast_index)
                forecast_df['forecast'] = forecast_values
            
            return {
                "model_type": "lstm",
                "forecast": forecast_df
            }
        
        except Exception as e:
            return f"Error forecasting with LSTM model: {str(e)}"
    
    def _create_sequences(self, data, seq_length, return_y=False):
        """Create sequences for LSTM training"""
        X = []
        y = []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        if return_y:
            return np.array(X), np.array(y)
        else:
            return np.array(X)
