#!/usr/bin/env python3
"""
Configuration Manager for UAV Vision System
Centralized configuration management with validation
"""

import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from dataclasses import dataclass, field
from enum import Enum


class ConfigSource(Enum):
    """Configuration source types"""
    FILE = "file"
    PARAMETERS = "parameters"
    ENVIRONMENT = "environment"
    DEFAULT = "default"


@dataclass
class ConfigEntry:
    """Configuration entry with metadata"""
    key: str
    value: Any
    source: ConfigSource
    description: str = ""
    required: bool = False
    validator: Optional[callable] = None
    default: Any = None


class ConfigValidator:
    """Configuration validator"""
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate value type"""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate numeric range"""
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any]) -> bool:
        """Validate value is in choices"""
        return value in choices
    
    @staticmethod
    def validate_path(value: str, must_exist: bool = False) -> bool:
        """Validate file path"""
        path = Path(value)
        if must_exist:
            return path.exists()
        return True
    
    @staticmethod
    def validate_positive(value: float) -> bool:
        """Validate positive number"""
        return value > 0
    
    @staticmethod
    def validate_non_negative(value: float) -> bool:
        """Validate non-negative number"""
        return value >= 0


class ConfigManager:
    """
    Centralized configuration manager with validation.
    
    Supports:
    - Loading from YAML/JSON files
    - ROS2 parameters
    - Environment variables
    - Default values
    - Validation
    - Type checking
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self._config: Dict[str, Any] = {}
        self._entries: Dict[str, ConfigEntry] = {}
        self._validation_errors: List[str] = []
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to YAML or JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")
            
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    self._config = yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    self._config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            return True
        except Exception as e:
            print(f"Error loading config file: {e}")
            return False
    
    def load_from_ros2_params(self, node) -> bool:
        """
        Load configuration from ROS2 node parameters.
        
        Args:
            node: ROS2 node instance
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Get all parameter names
            param_names = node._parameters.keys()
            for param_name in param_names:
                param_value = node.get_parameter(param_name).value
                self._config[param_name] = param_value
            return True
        except Exception as e:
            print(f"Error loading ROS2 parameters: {e}")
            return False
    
    def load_from_environment(self, prefix: str = "UAV_") -> bool:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variable names
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower()
                    # Try to parse as JSON, fallback to string
                    try:
                        self._config[config_key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        self._config[config_key] = value
            return True
        except Exception as e:
            print(f"Error loading environment variables: {e}")
            return False
    
    def register_entry(
        self,
        key: str,
        default: Any = None,
        description: str = "",
        required: bool = False,
        validator: Optional[callable] = None,
        source: ConfigSource = ConfigSource.DEFAULT
    ):
        """
        Register a configuration entry.
        
        Args:
            key: Configuration key
            default: Default value
            description: Description of the entry
            required: Whether this entry is required
            validator: Validation function
            source: Configuration source
        """
        entry = ConfigEntry(
            key=key,
            value=self._config.get(key, default),
            source=source,
            description=description,
            required=required,
            validator=validator,
            default=default
        )
        self._entries[key] = entry
        
        # Set default if not in config
        if key not in self._config and default is not None:
            self._config[key] = default
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def validate(self) -> bool:
        """
        Validate all registered configuration entries.
        
        Returns:
            True if all validations pass, False otherwise
        """
        self._validation_errors = []
        
        for key, entry in self._entries.items():
            # Check required
            if entry.required and key not in self._config:
                self._validation_errors.append(
                    f"Required config '{key}' is missing"
                )
                continue
            
            # Skip validation if value not set
            if key not in self._config:
                continue
            
            value = self._config[key]
            
            # Run validator if provided
            if entry.validator:
                try:
                    if not entry.validator(value):
                        self._validation_errors.append(
                            f"Config '{key}' validation failed: {value}"
                        )
                except Exception as e:
                    self._validation_errors.append(
                        f"Config '{key}' validation error: {e}"
                    )
        
        return len(self._validation_errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """
        Get validation errors.
        
        Returns:
            List of validation error messages
        """
        return self._validation_errors.copy()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()
    
    def merge(self, other_config: Dict[str, Any], overwrite: bool = True):
        """
        Merge another configuration dictionary.
        
        Args:
            other_config: Configuration dictionary to merge
            overwrite: Whether to overwrite existing values
        """
        if overwrite:
            self._config.update(other_config)
        else:
            for key, value in other_config.items():
                if key not in self._config:
                    self._config[key] = value
    
    def save_to_file(self, file_path: str, format: str = "yaml") -> bool:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to output file
            format: File format ("yaml" or "json")
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            path = Path(file_path)
            with open(path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(self._config, f, default_flow_style=False)
                elif format.lower() == "json":
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            print(f"Error saving config file: {e}")
            return False
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a configuration entry.
        
        Args:
            key: Configuration key
            
        Returns:
            Dictionary with entry information or None
        """
        if key not in self._entries:
            return None
        
        entry = self._entries[key]
        return {
            "key": entry.key,
            "value": self._config.get(key),
            "default": entry.default,
            "source": entry.source.value,
            "description": entry.description,
            "required": entry.required
        }

