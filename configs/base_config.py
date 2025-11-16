"""Base configuration loader for ECG digitization project."""

import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads base_config.yaml.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'base.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle config inheritance
    if '_BASE_' in config:
        base_config_path = os.path.join(os.path.dirname(__file__), config['_BASE_'])
        base_config = load_config(base_config_path)

        # Merge configurations (current config overrides base)
        merged_config = {**base_config}
        merged_config.update(config)
        merged_config.pop('_BASE_')  # Remove the inheritance marker

        return merged_config

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_stage_config(stage: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get stage-specific configuration.

    Args:
        stage: Stage name ('stage0', 'stage1', 'stage2')
        config: Base configuration

    Returns:
        Stage-specific configuration
    """
    if config is None:
        config = load_config()

    stage_config_file = os.path.join(os.path.dirname(__file__), f'{stage}_config.yaml')

    if os.path.exists(stage_config_file):
        stage_config = load_config(stage_config_file)
        # Merge with base config
        merged_config = {**config}
        merged_config.update(stage_config)
        return merged_config

    # Return base config if stage-specific config doesn't exist
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'DEVICE', 'TRAIN', 'DATA', 'MODEL', 'COMPETITION'
    ]

    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False

    return True


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"  Device: {config.get('DEVICE', {}).get('DEVICE', 'cpu')}")
    print(f"  Competition Mode: {config.get('COMPETITION', {}).get('MODE', 'unknown')}")
    print(f"  Data Root: {config.get('PATHS', {}).get('DATA_ROOT', './data')}")