#!/usr/bin/env python3
"""
Plugin System for UAV Vision
Provides extensible plugin architecture for SLAM backends and other components
"""

import importlib
import importlib.util
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class PluginInfo:
    """Plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    class_name: str
    module_path: str


class PluginInterface(ABC):
    """
    Base interface for all plugins.
    All plugins must inherit from this class.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup plugin resources"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.
        
        Returns:
            Dictionary with plugin information
        """
        pass


class SLAMBackendInterface(PluginInterface):
    """
    Interface for SLAM backend plugins.
    """
    
    @abstractmethod
    def process_frame(self, image, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Process a frame for SLAM.
        
        Args:
            image: Input image (numpy array)
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with pose, map, or None if processing failed
        """
        pass
    
    @abstractmethod
    def get_map(self) -> Optional[Any]:
        """
        Get current map.
        
        Returns:
            Map object or None
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset SLAM system"""
        pass


class PathPlannerInterface(PluginInterface):
    """
    Interface for path planner plugins.
    """
    
    @abstractmethod
    def plan_path(
        self,
        start: List[float],
        goal: List[float],
        obstacles: Optional[Any] = None
    ) -> Optional[List[List[float]]]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            obstacles: Obstacle information
            
        Returns:
            List of waypoints or None if planning failed
        """
        pass
    
    @abstractmethod
    def update_map(self, map_data: Any):
        """
        Update the map used for planning.
        
        Args:
            map_data: Map data
        """
        pass


class PluginLoader:
    """
    Plugin loader for dynamic plugin loading.
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize plugin loader.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: Dict[str, Type[PluginInterface]] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin_dir(self, directory: str):
        """
        Register a directory to search for plugins.
        
        Args:
            directory: Directory path
        """
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
    
    def load_plugin(
        self,
        module_path: str,
        class_name: str,
        plugin_name: Optional[str] = None
    ) -> Optional[Type[PluginInterface]]:
        """
        Load a plugin from module.
        
        Args:
            module_path: Path to module (e.g., 'plugins.my_plugin')
            class_name: Name of plugin class
            plugin_name: Optional plugin name (defaults to class_name)
            
        Returns:
            Plugin class or None if loading failed
        """
        try:
            # Try importing from installed packages first
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            # Verify it's a plugin
            if not issubclass(plugin_class, PluginInterface):
                self.logger.error(
                    f"{class_name} does not implement PluginInterface"
                )
                return None
            
            name = plugin_name or class_name
            self.loaded_plugins[name] = plugin_class
            self.logger.info(f"Loaded plugin: {name}")
            
            return plugin_class
            
        except ImportError as e:
            self.logger.error(f"Failed to import plugin {module_path}: {e}")
            return None
        except AttributeError as e:
            self.logger.error(
                f"Class {class_name} not found in {module_path}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Error loading plugin: {e}", exc_info=True)
            return None
    
    def load_plugin_from_file(
        self,
        file_path: str,
        class_name: str,
        plugin_name: Optional[str] = None
    ) -> Optional[Type[PluginInterface]]:
        """
        Load plugin from file path.
        
        Args:
            file_path: Path to Python file
            class_name: Name of plugin class
            plugin_name: Optional plugin name
            
        Returns:
            Plugin class or None
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.error(f"Plugin file not found: {file_path}")
                return None
            
            # Create module spec
            spec = importlib.util.spec_from_file_location(
                file_path_obj.stem,
                file_path
            )
            
            if spec is None or spec.loader is None:
                self.logger.error(f"Failed to create spec for {file_path}")
                return None
            
            # Load module
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path_obj.stem] = module
            spec.loader.exec_module(module)
            
            # Get plugin class
            plugin_class = getattr(module, class_name)
            
            if not issubclass(plugin_class, PluginInterface):
                self.logger.error(
                    f"{class_name} does not implement PluginInterface"
                )
                return None
            
            name = plugin_name or class_name
            self.loaded_plugins[name] = plugin_class
            self.logger.info(f"Loaded plugin from file: {name}")
            
            return plugin_class
            
        except Exception as e:
            self.logger.error(
                f"Error loading plugin from file {file_path}: {e}",
                exc_info=True
            )
            return None
    
    def load_plugins_from_directory(self, directory: str) -> List[str]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            List of loaded plugin names
        """
        loaded = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.warning(f"Plugin directory not found: {directory}")
            return loaded
        
        # Look for Python files
        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            # Try to load as plugin
            # Convention: class name should match file name (capitalized)
            class_name = file_path.stem.replace("_", " ").title().replace(" ", "")
            
            plugin_class = self.load_plugin_from_file(
                str(file_path),
                class_name
            )
            
            if plugin_class:
                loaded.append(class_name)
        
        return loaded
    
    def create_plugin(
        self,
        plugin_name: str,
        config: Dict[str, Any]
    ) -> Optional[PluginInterface]:
        """
        Create an instance of a loaded plugin.
        
        Args:
            plugin_name: Name of the plugin
            config: Configuration dictionary
            
        Returns:
            Plugin instance or None
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.error(f"Plugin not found: {plugin_name}")
            return None
        
        try:
            plugin_class = self.loaded_plugins[plugin_name]
            instance = plugin_class()
            
            if instance.initialize(config):
                return instance
            else:
                self.logger.error(f"Failed to initialize plugin: {plugin_name}")
                return None
                
        except Exception as e:
            self.logger.error(
                f"Error creating plugin instance {plugin_name}: {e}",
                exc_info=True
            )
            return None
    
    def list_plugins(self) -> List[str]:
        """
        List all loaded plugins.
        
        Returns:
            List of plugin names
        """
        return list(self.loaded_plugins.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """
        Get plugin information.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            PluginInfo or None
        """
        return self.plugin_info.get(plugin_name)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if unloaded, False otherwise
        """
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]
            if plugin_name in self.plugin_info:
                del self.plugin_info[plugin_name]
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        return False


class PluginManager:
    """
    Plugin manager for managing plugin lifecycle.
    """
    
    def __init__(self):
        """Initialize plugin manager"""
        self.loader = PluginLoader()
        self.active_plugins: Dict[str, PluginInterface] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin_dir(self, directory: str):
        """Register plugin directory"""
        self.loader.register_plugin_dir(directory)
    
    def load_plugin(
        self,
        plugin_name: str,
        module_path: str,
        class_name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Load and activate a plugin.
        
        Args:
            plugin_name: Plugin name
            module_path: Module path
            class_name: Class name
            config: Configuration
            
        Returns:
            True if successful, False otherwise
        """
        plugin_class = self.loader.load_plugin(module_path, class_name, plugin_name)
        if not plugin_class:
            return False
        
        instance = self.loader.create_plugin(plugin_name, config)
        if not instance:
            return False
        
        self.active_plugins[plugin_name] = instance
        self.logger.info(f"Activated plugin: {plugin_name}")
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Get active plugin instance.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self.active_plugins.get(plugin_name)
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate a plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if deactivated, False otherwise
        """
        if plugin_name in self.active_plugins:
            plugin = self.active_plugins[plugin_name]
            plugin.cleanup()
            del self.active_plugins[plugin_name]
            self.logger.info(f"Deactivated plugin: {plugin_name}")
            return True
        return False
    
    def list_active_plugins(self) -> List[str]:
        """List active plugins"""
        return list(self.active_plugins.keys())
    
    def cleanup_all(self):
        """Cleanup all active plugins"""
        for plugin_name, plugin in list(self.active_plugins.items()):
            self.deactivate_plugin(plugin_name)

