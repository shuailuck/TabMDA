"""
Configuration for the TabPFN weights used by TabMDA.

Read from ``tabpfn_config.json`` (next to this file), or from the path given in
the ``TABPFN_CONFIG`` environment variable. Supported fields:

    model_path : optional path to a local TabPFN checkpoint (``.cpkt``). When set,
                 the checkpoint is loaded from this file and no download happens.
    model_url  : URL used to download the TabPFN v1 checkpoint when it is not
                 already present locally.

The default ``model_url`` points at a HuggingFace mirror of the TabPFN v1
checkpoint, because the original GitHub URL no longer exists.
"""

import json
import os

_DEFAULT_MODEL_URL = (
    "https://huggingface.co/spaces/TabPFN/TabPFNPrediction/resolve/main/"
    "TabPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
)

_DEFAULTS = {
    "model_path": None,
    "model_url": _DEFAULT_MODEL_URL,
}

_HERE = os.path.dirname(os.path.abspath(__file__))

_CONFIG_PATH = os.environ.get(
    "TABPFN_CONFIG", os.path.join(_HERE, "tabpfn_config.json")
)


def load_tabpfn_config():
    """Return the merged configuration dict (defaults overlaid with the file)."""
    config = dict(_DEFAULTS)
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            if isinstance(user_config, dict):
                config.update(user_config)
        except (OSError, ValueError) as exc:
            print(f"[tabpfn_config] Could not read {_CONFIG_PATH}: {exc}")
    return config
