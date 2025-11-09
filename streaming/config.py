"""
Configuration management for streaming system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader and manager."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML config file
        """
        if config_file is None:
            # Default config file
            config_file = Path(__file__).parent.parent / "config" / "streaming.yaml"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            # Return default configuration
            return self._default_config()

        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'app': {
                'name': 'OptionGreeksStreaming',
                'version': '3.1.0',
                'log_level': 'INFO'
            },
            'processing': {
                'batch_size': 500,
                'window_ms': 50,
                'interest_rate': 5.0,
                'use_iv_smoothing': True,
                'smoothing_alpha': 0.3
            },
            'state': {
                'cache_size': 10000,
                'iv_history_depth': 100,
                'state_ttl_days': 7
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'sources': {
                'redis_streams': {'enabled': True},
                'csv': {'enabled': False}
            },
            'output': {
                'redis_streams': {'enabled': True}
            }
        }

    def get(self, key: str, default=None):
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'redis.host')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.get(key)
