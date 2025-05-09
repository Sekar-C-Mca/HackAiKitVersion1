import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigManager:
    """
    Configuration manager for HackAI-Kit.
    
    Handles loading configuration from environment variables, .env files,
    and JSON configuration files. Manages API keys and other settings
    for various modules.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_file: Optional path to a JSON configuration file
        """
        # Load environment variables from .env file if present
        load_dotenv()
        
        # Initialize config dict with default values
        self.config = {
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "google": os.environ.get("GOOGLE_API_KEY", ""),
                "huggingface": os.environ.get("HUGGINGFACE_API_KEY", ""),
            },
            "model_defaults": {
                "openai": "gpt-3.5-turbo",
                "gemini": "gemini-pro",
                "huggingface": {},
            },
            "storage": {
                "models_dir": "models",
                "data_dir": "data",
                "temp_dir": "temp",
            },
            "logging": {
                "level": "INFO",
                "file": "hackaikit.log",
            },
            "ui": {
                "theme": "light",
                "colormap": "viridis",
            },
        }
        
        # Load from config file if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Warning: Config file {config_file} not found.")
            return
            
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Merge with existing config
            self._merge_configs(self.config, file_config)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading config from {config_file}: {str(e)}")
    
    def _merge_configs(self, base_config: Dict, new_config: Dict) -> None:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration to update
            new_config: New configuration to merge in
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_config(self, config_file: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_file: Path to save the configuration file
        """
        config_path = Path(config_file)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write config to file
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Saved configuration to {config_file}")
        except Exception as e:
            print(f"Error saving config to {config_file}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'api_keys.openai')
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'api_keys.openai')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the right level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'google', 'huggingface')
            
        Returns:
            API key string or None if not found
        """
        return self.get(f"api_keys.{provider.lower()}")
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """
        Set an API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'google', 'huggingface')
            api_key: API key string
        """
        self.set(f"api_keys.{provider.lower()}", api_key)
        
    def get_default_model(self, provider: str) -> Optional[str]:
        """
        Get the default model for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
            
        Returns:
            Default model name or None if not found
        """
        return self.get(f"model_defaults.{provider.lower()}")
    
    def get_models_dir(self) -> str:
        """
        Get the directory for storing models.
        
        Returns:
            Directory path string
        """
        return self.get("storage.models_dir", "models")

# Initialize a global config manager (can be imported elsewhere)
# config_manager = ConfigManager()