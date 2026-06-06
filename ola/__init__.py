"""Ordinal Lens Alignment (OLA): a frozen-backbone, inference-time method that
reads probe-recoverable ordinal evidence from mid-to-deep decoder layers and
corrects only the digit-token logits at the answer slot.

Flat module layout:
    ola.lens        Eq. 4 W_S-anchored lens, Eq. 6 multi-layer fusion, frozen-W_S checks
    ola.train       records, forward hooks, hidden-state cache, Stage A/B training, P^OLA cache
    ola.alignment   tokenizer id resolution, Eq. 7 correction, lambda* search, online processor
    ola.metrics     accuracy, MAE, SRCC
"""

from .lens import (
    LowRankOrdinalLens,
    MultiLayerOrdinalLens,
    sha256_tensor,
)
from .train import (
    OLAConfig,
    TrainConfig,
    TrainLog,
    cache_p_ola,
    collect_last_token_hidden_cache,
    extract_last_token_hidden,
    prepare_ordinal_records,
    register_last_token_hooks,
    remove_hooks,
    train_ola_lens,
)
from .alignment import (
    OLAAdaptiveProcessor,
    PAPER_LAMBDA_GRID,
    confidence,
    correct_digit_logits,
    default_grid,
    encode_without_special_tokens,
    resolve_digit_and_eos_ids,
    resolve_digit_token_ids,
    resolve_eos_token_ids,
    search_lambda,
    simulate_ala,
)
from .metrics import accuracy, mae, srcc

__version__ = "0.1.0"

__all__ = [
    # ola.lens
    "LowRankOrdinalLens",
    "MultiLayerOrdinalLens",
    "sha256_tensor",
    # ola.train
    "OLAConfig",
    "TrainConfig",
    "TrainLog",
    "cache_p_ola",
    "collect_last_token_hidden_cache",
    "extract_last_token_hidden",
    "prepare_ordinal_records",
    "register_last_token_hooks",
    "remove_hooks",
    "train_ola_lens",
    # ola.alignment
    "OLAAdaptiveProcessor",
    "PAPER_LAMBDA_GRID",
    "confidence",
    "correct_digit_logits",
    "default_grid",
    "encode_without_special_tokens",
    "resolve_digit_and_eos_ids",
    "resolve_digit_token_ids",
    "resolve_eos_token_ids",
    "search_lambda",
    "simulate_ala",
    # ola.metrics
    "accuracy",
    "mae",
    "srcc",
    "__version__",
]
