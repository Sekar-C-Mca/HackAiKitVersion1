import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from hackaikit.core.base_module import BaseModule


class ModuleManager:
    """
    Module manager for discovering and loading plugin modules dynamically.
    Implements the plugin architecture described in the design document.
    """

    def __init__(self, modules_package="hackaikit.modules", config_manager=None):
        """
        Initialize the module manager.
        
        Args:
            modules_package: The package containing plugin modules
            config_manager: Configuration manager to pass to modules
        """
        self.modules_package = modules_package
        self.config_manager = config_manager
        self.available_modules: Dict[str, Type[BaseModule]] = {}
        self.loaded_instances: Dict[str, BaseModule] = {}
        
        # Discover available modules on initialization
        self.discover_modules()
        
    def discover_modules(self) -> List[str]:
        """
        Discover available modules by scanning the modules directory.
        
        Returns:
            List of discovered module names
        """
        self.available_modules = {}
        
        # Get the modules package
        try:
            package = importlib.import_module(self.modules_package)
        except ImportError:
            print(f"Error: Could not import modules package {self.modules_package}")
            return []
            
        # Get the directory path for the package
        package_path = Path(package.__path__[0])
        
        # Scan for module files/packages
        for module_info in pkgutil.iter_modules([str(package_path)]):
            module_name = module_info.name
            
            # Skip __init__.py and base_module.py
            if module_name.startswith('__') or module_name == 'base_module':
                continue
                
            try:
                # Import the module
                module = importlib.import_module(f"{self.modules_package}.{module_name}")
                
                # Look for classes that inherit from BaseModule
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseModule) and obj != BaseModule and 
                        obj.__module__ == f"{self.modules_package}.{module_name}"):
                        self.available_modules[name] = obj
                        print(f"Discovered module: {name}")
            except Exception as e:
                print(f"Error loading module {module_name}: {str(e)}")
        
        return list(self.available_modules.keys())
    
    def get_module(self, module_name: str) -> BaseModule:
        """
        Get an instance of a module by name.
        
        Args:
            module_name: Name of the module to get
            
        Returns:
            Instance of the module
        
        Raises:
            ValueError: If the module is not found
        """
        # Check if module is already instantiated
        if module_name in self.loaded_instances:
            return self.loaded_instances[module_name]
        
        # Check if module is available
        if module_name not in self.available_modules:
            raise ValueError(f"Module '{module_name}' not found")
        
        # Instantiate the module
        module_class = self.available_modules[module_name]
        module_instance = module_class(config_manager=self.config_manager)
        
        # Cache the instance
        self.loaded_instances[module_name] = module_instance
        
        return module_instance
    
    def get_all_modules(self) -> Dict[str, BaseModule]:
        """
        Get instances of all available modules.
        
        Returns:
            Dictionary of module instances
        """
        # Instantiate any modules that haven't been loaded yet
        for module_name in self.available_modules:
            if module_name not in self.loaded_instances:
                self.get_module(module_name)
                
        return self.loaded_instances
    
    def get_module_info(self) -> List[Dict]:
        """
        Get information about all available modules.
        
        Returns:
            List of module information dictionaries
        """
        module_info = []
        
        for module_name, module_class in self.available_modules.items():
            # Get basic info without instantiating
            info = {
                "name": module_name,
                "description": module_class.__doc__ or "No description available."
            }
            module_info.append(info)
            
        return module_info 