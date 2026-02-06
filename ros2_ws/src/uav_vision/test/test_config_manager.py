"""
Unit tests for ConfigManager
"""

import pytest
import yaml
import json
import tempfile
import os
from pathlib import Path

from uav_vision.config_manager import (
    ConfigManager,
    ConfigSource,
    ConfigValidator
)


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def test_init_empty(self):
        """Test initialization with no config file"""
        config = ConfigManager()
        assert config._config == {}
        assert len(config._entries) == 0
    
    def test_load_from_yaml_file(self, temp_config_file):
        """Test loading configuration from YAML file"""
        config = ConfigManager()
        result = config.load_from_file(temp_config_file)
        
        assert result is True
        assert config.get('test_key') == 'test_value'
        assert config.get('numeric_value') == 42
        assert config.get('list_value') == [1, 2, 3]
    
    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file"""
        config_file = tmp_path / "test_config.json"
        config_data = {
            'test_key': 'test_value',
            'numeric_value': 42
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = ConfigManager()
        result = config.load_from_file(str(config_file))
        
        assert result is True
        assert config.get('test_key') == 'test_value'
        assert config.get('numeric_value') == 42
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        config = ConfigManager()
        result = config.load_from_file('/nonexistent/file.yaml')
        
        assert result is False
    
    def test_register_entry(self):
        """Test registering configuration entries"""
        config = ConfigManager()
        
        config.register_entry(
            key='test_key',
            default='default_value',
            description='Test description',
            required=False
        )
        
        assert 'test_key' in config._entries
        entry = config._entries['test_key']
        assert entry.key == 'test_key'
        assert entry.default == 'default_value'
        assert entry.description == 'Test description'
        assert entry.required is False
    
    def test_get_with_default(self):
        """Test getting value with default"""
        config = ConfigManager()
        value = config.get('nonexistent_key', 'default')
        
        assert value == 'default'
    
    def test_get_existing_value(self):
        """Test getting existing value"""
        config = ConfigManager()
        config.set('test_key', 'test_value')
        
        value = config.get('test_key')
        assert value == 'test_value'
    
    def test_set_value(self):
        """Test setting configuration value"""
        config = ConfigManager()
        config.set('test_key', 'test_value')
        
        assert config.get('test_key') == 'test_value'
    
    def test_validation_success(self):
        """Test successful validation"""
        config = ConfigManager()
        
        config.register_entry(
            key='positive_number',
            default=10.0,
            validator=lambda x: ConfigValidator.validate_positive(x)
        )
        config.set('positive_number', 5.0)
        
        result = config.validate()
        assert result is True
        assert len(config.get_validation_errors()) == 0
    
    def test_validation_failure(self):
        """Test validation failure"""
        config = ConfigManager()
        
        config.register_entry(
            key='positive_number',
            default=10.0,
            validator=lambda x: ConfigValidator.validate_positive(x)
        )
        config.set('positive_number', -5.0)  # Invalid: negative
        
        result = config.validate()
        assert result is False
        assert len(config.get_validation_errors()) > 0
    
    def test_required_entry_missing(self):
        """Test validation with missing required entry"""
        config = ConfigManager()
        
        config.register_entry(
            key='required_key',
            required=True
        )
        
        result = config.validate()
        assert result is False
        errors = config.get_validation_errors()
        assert any('required' in error.lower() for error in errors)
    
    def test_merge_config(self):
        """Test merging configurations"""
        config1 = ConfigManager()
        config1.set('key1', 'value1')
        config1.set('key2', 'value2')
        
        config2 = {'key2': 'new_value2', 'key3': 'value3'}
        
        config1.merge(config2, overwrite=True)
        
        assert config1.get('key1') == 'value1'
        assert config1.get('key2') == 'new_value2'  # Overwritten
        assert config1.get('key3') == 'value3'
    
    def test_merge_no_overwrite(self):
        """Test merging without overwriting"""
        config1 = ConfigManager()
        config1.set('key1', 'value1')
        
        config2 = {'key1': 'new_value1', 'key2': 'value2'}
        
        config1.merge(config2, overwrite=False)
        
        assert config1.get('key1') == 'value1'  # Not overwritten
        assert config1.get('key2') == 'value2'
    
    def test_save_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file"""
        config = ConfigManager()
        config.set('test_key', 'test_value')
        config.set('numeric', 42)
        
        output_file = tmp_path / "output.yaml"
        result = config.save_to_file(str(output_file), format='yaml')
        
        assert result is True
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)
            assert loaded['test_key'] == 'test_value'
            assert loaded['numeric'] == 42
    
    def test_save_to_json(self, tmp_path):
        """Test saving configuration to JSON file"""
        config = ConfigManager()
        config.set('test_key', 'test_value')
        
        output_file = tmp_path / "output.json"
        result = config.save_to_file(str(output_file), format='json')
        
        assert result is True
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            loaded = json.load(f)
            assert loaded['test_key'] == 'test_value'
    
    def test_get_entry_info(self):
        """Test getting entry information"""
        config = ConfigManager()
        config.register_entry(
            key='test_key',
            default='default_value',
            description='Test description',
            required=True
        )
        config.set('test_key', 'actual_value')
        
        info = config.get_entry_info('test_key')
        
        assert info is not None
        assert info['key'] == 'test_key'
        assert info['value'] == 'actual_value'
        assert info['default'] == 'default_value'
        assert info['description'] == 'Test description'
        assert info['required'] is True
    
    def test_get_entry_info_nonexistent(self):
        """Test getting info for non-existent entry"""
        config = ConfigManager()
        info = config.get_entry_info('nonexistent')
        
        assert info is None


class TestConfigValidator:
    """Test suite for ConfigValidator"""
    
    def test_validate_type(self):
        """Test type validation"""
        assert ConfigValidator.validate_type(5, int) is True
        assert ConfigValidator.validate_type(5.0, float) is True
        assert ConfigValidator.validate_type("string", str) is True
        assert ConfigValidator.validate_type(5, str) is False
    
    def test_validate_range(self):
        """Test range validation"""
        assert ConfigValidator.validate_range(5.0, 0.0, 10.0) is True
        assert ConfigValidator.validate_range(15.0, 0.0, 10.0) is False
        assert ConfigValidator.validate_range(-5.0, 0.0, 10.0) is False
    
    def test_validate_choice(self):
        """Test choice validation"""
        choices = ['option1', 'option2', 'option3']
        assert ConfigValidator.validate_choice('option1', choices) is True
        assert ConfigValidator.validate_choice('invalid', choices) is False
    
    def test_validate_positive(self):
        """Test positive number validation"""
        assert ConfigValidator.validate_positive(5.0) is True
        assert ConfigValidator.validate_positive(0.0) is False
        assert ConfigValidator.validate_positive(-5.0) is False
    
    def test_validate_non_negative(self):
        """Test non-negative validation"""
        assert ConfigValidator.validate_non_negative(5.0) is True
        assert ConfigValidator.validate_non_negative(0.0) is True
        assert ConfigValidator.validate_non_negative(-5.0) is False

