"""utils.load_config
the training entrypoint should not hand-pick individual keys from the YAML because that doesn't scale as the config grows.
Instead, `load_config()` returns a normalized config with defaults and any
lightweight derivations applied (e.g., optional val path).
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def _normalize_config(raw: Dict[str, Any], *, config_dir: Path) -> Dict[str, Any]:
    """Apply defaults and small derivations.

    The output is expected to be directly passable to constructors:
    - `config['data']` -> `create_dataloaders(**...)`
    - `config['model'] | config['training']` -> `FishModule(...)`
    - `config['trainer']` -> `L.Trainer(...)`
    """
    raw = raw or {}

    defaults: Dict[str, Any] = {
        'seed': 42,
        'data': {
            'train_path': 'data/train.pt',
            'val_path': 'data/val.pt',
            'batch_size': 4,
            'num_workers': 4,
        },
        'model': {
            'in_channels': 1,
            'out_channels': 1,
            'features': [64, 128, 256, 512],
            'trilinear': False,
            'use_batchnorm': True,
            'loss_type': 'mse',
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'early_stopping': True,
            'patience': 10,
            'deterministic': False,
        },
        'trainer': {
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': '32',
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 1,
            'log_every_n_steps': 10,
        },
        'checkpoint': {
            'dir': 'checkpoints',
            'save_top_k': 3,
        },
        'logging': {
            'dir': 'logs',
            'experiment_name': 'zebrafish_unet',
        },
    }

    config = merge_configs(defaults, raw)

    # Paths: allow relative paths in YAML (relative to repo root / current cwd).
    # Keep behavior simple + stable; do not auto-expand into absolutes.
    data_cfg = config.get('data', {})
    val_path = data_cfg.get('val_path')
    if val_path is not None and not Path(val_path).exists():
        # Driver previously conditionally passed None when val file doesn't exist.
        data_cfg['val_path'] = None

    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        raw = yaml.safe_load(f)

    return _normalize_config(raw, config_dir=config_file.parent)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
