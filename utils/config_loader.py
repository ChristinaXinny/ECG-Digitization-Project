"""Configuration loader utility for ECG Digitization Project."""

import os
import yaml
import json
from typing import Dict, Any, Union
from pathlib import Path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save based on file extension
    if save_path.suffix.lower() in ['.yaml', '.yml']:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif save_path.suffix.lower() == '.json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {save_path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    import copy

    result = copy.deepcopy(base_config)

    def _merge_dict(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dict(base[key], value)
            else:
                base[key] = value

    _merge_dict(result, override_config)
    return result