#!/usr/bin/env python
"""
Test script for the ModuleManager to ensure proper module discovery and loading.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from hackaikit.core.config_manager import ConfigManager
from hackaikit.core.module_manager import ModuleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("module-manager-test")

def main():
    """Main test function"""
    logger.info("Testing ModuleManager...")
    
    # Initialize the config manager
    logger.info("Initializing ConfigManager...")
    config_manager = ConfigManager()
    
    # Initialize the module manager
    logger.info("Initializing ModuleManager...")
    module_manager = ModuleManager(config_manager=config_manager)
    
    # Discover available modules
    logger.info("Discovering modules...")
    available_modules = module_manager.discover_modules()
    logger.info(f"Found {len(available_modules)} modules: {available_modules}")
    
    # Get module info
    module_info = module_manager.get_module_info()
    logger.info("Module information:")
    for info in module_info:
        logger.info(f"  - {info['name']}: {info['description']}")
    
    # Test loading each module
    logger.info("Testing module loading...")
    for module_name in available_modules:
        try:
            module = module_manager.get_module(module_name)
            logger.info(f"  - Successfully loaded {module_name}")
            logger.info(f"    Module info: {module.get_info()}")
        except Exception as e:
            logger.error(f"  - Failed to load {module_name}: {str(e)}")
    
    logger.info("ModuleManager test completed.")

if __name__ == "__main__":
    main() 