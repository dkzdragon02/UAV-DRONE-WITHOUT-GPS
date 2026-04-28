import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from dataclasses import dataclass, field
from enum import Enum

class ConfigSource(Enum):
    FILE = "file"
    PARAMETERS = "parameters"
    ENVIRONMENT = "environment"
    DEFAULT = "default"

@dataclass
class ConfigEntry:
    key: str
    value: Any
    source: ConfigSource
    description: str = ""
    required: bool = False
    validator: Optional[callable] = None
    default: Any = None

class ConfigValidator:
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any]) -> bool:
        return value in choices
    
    @staticmethod
    def validate_path(value: str, must_exist: bool = False) -> bool:
        path = Path(value)
        if must_exist:
            return path.exists()
        return True
    
    @staticmethod
    def validate_positive(value: float) -> bool:
        return value > 0
    
    @staticmethod
    def validate_non_negative(value: float) -> bool:
        return value >= 0


class ConfigManager:    
    def __init__(self, config_file: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._entries: Dict[str, ConfigEntry] = {}
        self._validation_errors: List[str] = []
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, file_path: str) -> bool:
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
        try:
            param_names = node._parameters.keys()
            for param_name in param_names:
                param_value = node.get_parameter(param_name).value
                self._config[param_name] = param_value
            return True
        except Exception as e:
            print(f"Error loading ROS2 parameters: {e}")
            return False
    
    def load_from_environment(self, prefix: str = "UAV_") -> bool:
        try:
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower()
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
        
        if key not in self._config and default is not None:
            self._config[key] = default
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        self._config[key] = value
    
    def validate(self) -> bool:
        self._validation_errors = [] 
        for key, entry in self._entries.items():
            if entry.required and key not in self._config:
                self._validation_errors.append(
                    f"Required config '{key}' is missing"
                )
                continue

            if key not in self._config:
                continue
            
            value = self._config[key]
            
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
        return self._validation_errors.copy()
    
    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()
    
    def merge(self, other_config: Dict[str, Any], overwrite: bool = True):
        if overwrite:
            self._config.update(other_config)
        else:
            for key, value in other_config.items():
                if key not in self._config:
                    self._config[key] = value
    
    def save_to_file(self, file_path: str, format: str = "yaml") -> bool:
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

