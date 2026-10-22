"""Default configuration for the TabMDA contrastive extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContrastiveConfig:
    # Context
    K: int = 2                                        # contexts per original sample
    context_size: Optional[float] = None              # None=whole pool, int=abs, float=proportion
    seed: int = 42

    # Adapter / model
    adapter_hid_dim: int = 256
    adapter_out_dim: int = 128
    dropout: float = 0.0
    use_projection: bool = True                       # decouple CE (r) from SupCon (z=g(r)) representations
    proj_dim: int = 128

    # Loss
    temperature: float = 0.2
    alpha: float = 0.5
    lambda_con: float = 0.1
    lambda_ctx: float = 0.0
    contrastive_mode: str = "grouped"                 # "grouped" | "uniform"
    require_negatives: bool = True
    con_warmup_ratio: float = 0.0                     # linear warmup over first fraction of steps

    # Training
    device: str = "cpu"
    optimizer: str = "adamw"                          # adamw | adam | sgd
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 30
    # Sampling the minibatch: number of distinct original samples. Must be >= 2.
    # None = use batch_size.
    n_samples_per_batch: Optional[int] = None

    # Evaluation
    K_eval: int = 2                                   # contexts used at eval (fixed protocol)
    aggregation: str = "probability_mean"             # "probability_mean" | "embedding_mean"

    # Backend / caching
    backend: str = "frozen_tabpfn"                    # "frozen_tabpfn" | "identity"
    cache_dir: Optional[str] = None
    model_fingerprint: Optional[str] = None

    # Logging
    log_interval: int = 10
    eval_interval: int = 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def fingerprint(self) -> str:
        import hashlib
        import json

        d = self.to_dict()
        d["backend"] = self.backend
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]