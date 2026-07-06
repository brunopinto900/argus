"""Loads yaml files from the repo's config/ directory."""

import os
import yaml

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))


def load_config(name: str = 'quadrotor.yaml') -> dict:
    with open(os.path.join(_CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def load_target_config() -> dict:
    return load_config('target.yaml')
