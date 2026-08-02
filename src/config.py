"""
CAMFS M1 — Configuration Management
===================================

Loads YAML configuration files and resolves inheritance/overrides.
Spec reference: §3 of research code standards.
"""

import argparse
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config file and resolve base inheritance if present."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = (config_path.parent / config.pop("_base_")).resolve()
        base_config = load_yaml(base_path)
        config = _deep_merge(base_config, config)

    return config

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment scripts."""
    parser = argparse.ArgumentParser(description="CAMFS M1 Federated Segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Federation partition seed (1103, 2207, 3301)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training seed (17, 29, 43)")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU index")
    return parser.parse_args()
