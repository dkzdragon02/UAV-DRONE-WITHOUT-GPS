#!/usr/bin/env python3

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
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    class_name: str
    module_path: str

class PluginInterface(ABC):
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

class SLAMBackendInterface(PluginInterface):  
    @abstractmethod
    def process_frame(self, image, timestamp: float) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_map(self) -> Optional[Any]:
        pass
    
    @abstractmethod
    def reset(self):
        pass

class PathPlannerInterface(PluginInterface):
    @abstractmethod
    def plan_path(
        self,
        start: List[float],
        goal: List[float],
        obstacles: Optional[Any] = None
    ) -> Optional[List[List[float]]]:
        pass
    
    @abstractmethod
    def update_map(self, map_data: Any):
        pass

class PluginLoader:
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: Dict[str, Type[PluginInterface]] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin_dir(self, directory: str):
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
    
    def load_plugin(
        self,
        module_path: str,
        class_name: str,
        plugin_name: Optional[str] = None
    ) -> Optional[Type[PluginInterface]]:
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
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
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.error(f"Plugin file not found: {file_path}")
                return None
            
            spec = importlib.util.spec_from_file_location(
                file_path_obj.stem,
                file_path
            )
            
            if spec is None or spec.loader is None:
                self.logger.error(f"Failed to create spec for {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path_obj.stem] = module
            spec.loader.exec_module(module)
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
        loaded = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.warning(f"Plugin directory not found: {directory}")
            return loaded

        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
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
        return list(self.loaded_plugins.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        return self.plugin_info.get(plugin_name)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]
            if plugin_name in self.plugin_info:
                del self.plugin_info[plugin_name]
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        return False

class PluginManager:
    def __init__(self):
        self.loader = PluginLoader()
        self.active_plugins: Dict[str, PluginInterface] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin_dir(self, directory: str):
        self.loader.register_plugin_dir(directory)
    
    def load_plugin(
        self,
        plugin_name: str,
        module_path: str,
        class_name: str,
        config: Dict[str, Any]
    ) -> bool:
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
        return self.active_plugins.get(plugin_name)
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        if plugin_name in self.active_plugins:
            plugin = self.active_plugins[plugin_name]
            plugin.cleanup()
            del self.active_plugins[plugin_name]
            self.logger.info(f"Deactivated plugin: {plugin_name}")
            return True
        return False
    
    def list_active_plugins(self) -> List[str]:
        return list(self.active_plugins.keys())
    
    def cleanup_all(self):
        for plugin_name, plugin in list(self.active_plugins.items()):
            self.deactivate_plugin(plugin_name)

