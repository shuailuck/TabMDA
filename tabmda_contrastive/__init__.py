"""TabMDA contrastive extension: frozen TabPFN embeddings + adapter + head.

Public API surface:
    - :class:`~tabmda_contrastive.config.ContrastiveConfig`
    - :class:`~tabmda_contrastive.model.ContrastiveModel`
    - :class:`~tabmda_contrastive.losses.JointLoss`, ``LossConfig``, ``ContrastiveLoss``
    - :func:`~tabmda_contrastive.context_sampler.sample_contexts`,
      :func:`~tabmda_contrastive.context_sampler.sample_eval_contexts`
    - :class:`~tabmda_contrastive.embedding_backend.FrozenTabPFNBackend`
    - :func:`~tabmda_contrastive.train.train`
    - :func:`~tabmda_contrastive.evaluate.evaluate_fixed_protocol`
"""

from .config import ContrastiveConfig

__all__ = ["ContrastiveConfig"]
__version__ = "0.1.0"
