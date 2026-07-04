"""Loads config/quadrotor.yaml from the repo root."""

import os
import yaml

_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'config', 'quadrotor.yaml')
)


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
