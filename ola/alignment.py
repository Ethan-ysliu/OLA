from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _ids_to_list(ids) -> List[int]:
    if ids is None:
        return []
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()
    if isinstance(ids, int):
        return [int(ids)]
    if isinstance(ids, (list, tuple)):
        if len(ids) == 1 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]
    return [int(ids)]


def encode_without_special_tokens(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return _ids_to_list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        return _ids_to_list(encoded.get("input_ids"))
    if hasattr(encoded, "input_ids"):
        return _ids_to_list(encoded.input_ids)
    return _ids_to_list(encoded)


def resolve_digit_token_ids(
    tokenizer,
    n_classes: int,
    digit_texts: Optional[Sequence[str]] = None,
    digit_templates: Sequence[str] = ("{digit}",),
) -> List[int]:
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    texts = list(digit_texts) if digit_texts is not None else [str(i) for i in range(n_classes)]
    if len(texts) != n_classes:
        raise ValueError("digit_texts length must match n_classes")

    digit_ids: List[int] = []
    for text in texts:
        tried = []
        chosen: Optional[int] = None
        for template in digit_templates:
            token_text = template.format(digit=text, label=text)
            ids = encode_without_special_tokens(tokenizer, token_text)
            tried.append((token_text, ids))
            if len(ids) == 1:
                chosen = int(ids[0])
                break
        if chosen is None:
            raise ValueError(f"label {text!r} is not single-token under templates: {tried}")
        digit_ids.append(chosen)

    if len(set(digit_ids)) != len(digit_ids):
        raise ValueError(f"digit token ids must be unique, got {digit_ids}")
    return digit_ids


def resolve_eos_token_ids(tokenizer, extra_eos_ids: Optional[Sequence[int]] = None) -> List[int]:
    eos_ids = set()
    for attr in ("eos_token_id", "eos_token_ids"):
        if hasattr(tokenizer, attr):
            eos_ids.update(_ids_to_list(getattr(tokenizer, attr)))
    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_ids and eos_token is not None:
        eos_ids.update(encode_without_special_tokens(tokenizer, eos_token))
    if extra_eos_ids is not None:
        eos_ids.update(int(x) for x in extra_eos_ids)
    return sorted(eos_ids)


def resolve_digit_and_eos_ids(
    tokenizer,
    n_classes: int,
    digit_texts: Optional[Sequence[str]] = None,
    digit_templates: Sequence[str] = ("{digit}",),
    extra_eos_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    digit_ids = resolve_digit_token_ids(tokenizer, n_classes, digit_texts, digit_templates)
    eos_ids = resolve_eos_token_ids(tokenizer, extra_eos_ids)
    overlap = set(digit_ids) & set(eos_ids)
    if overlap:
        raise ValueError(f"digit/EOS overlap would corrupt the invariant: {overlap}")
    return digit_ids, eos_ids


def confidence(P: torch.Tensor, mode: str = "max_prob") -> torch.Tensor:
    """Confidence gate omega(P^OLA). The paper uses max_prob = max_c P^OLA(c)."""
    if mode == "max_prob":
        return P.max(dim=-1, keepdim=True)[0]
    if mode == "entropy":
        ent = -(P * P.clamp(min=1e-12).log()).sum(dim=-1, keepdim=True)
        C = P.shape[-1]
        return 1.0 - ent / math.log(max(C, 2))
    raise ValueError(f"conf_mode must be 'max_prob' or 'entropy', got {mode!r}")


def correct_digit_logits(
    digit_logits: torch.Tensor,
    P_ola: torch.Tensor,
    lam: float,
    conf_mode: str = "max_prob",
) -> torch.Tensor:
    """Eq. 7 discrepancy correction, restricted to the digit-token logits.

        L'(c) = L_na(c) - lambda * omega(P^OLA) * (Q(c) - P^OLA(c)),

    with Q = softmax(L_na over S_digit) and omega = max_c P^OLA(c). The sign is
    such that a digit over-favoured by Q relative to P^OLA is pushed down.
    """
    if digit_logits.shape != P_ola.shape:
        raise ValueError(
            f"shape mismatch: digit_logits {tuple(digit_logits.shape)} "
            f"vs P_ola {tuple(P_ola.shape)}"
        )
    if lam == 0.0:
        return digit_logits
    conf = confidence(P_ola, conf_mode)
    Q = F.softmax(digit_logits, dim=-1)
    return digit_logits - float(lam) * conf * (Q - P_ola)


def simulate_ala(
    base_logits: torch.Tensor,
    P_target: torch.Tensor,
    lam: float,
    conf_mode: str = "max_prob",
) -> torch.Tensor:
    """Offline preview of the online prediction: argmax of the corrected digit logits."""
    return correct_digit_logits(base_logits, P_target, lam, conf_mode).argmax(dim=-1)


# Algorithm 2 denotes the validation search set for the discrepancy strength
# lambda* abstractly as Lambda. This release uses the paper-reproduction grid
# below as the default search set.
PAPER_LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)


def default_grid(n: int = 41, hi: float = 20.0) -> List[float]:
    """Uniform lambda grid of `n` points in [0, hi].

    This helper is provided for users who want a finer custom sweep; pass its
    output as the `grid` argument of `search_lambda`. It is NOT the default
    search set.
    """
    if n < 2:
        raise ValueError("grid size n must be at least 2")
    return [round(hi * i / (n - 1), 4) for i in range(n)]


def search_lambda(
    base_logits_val: torch.Tensor,
    P_val: torch.Tensor,
    y_val: torch.Tensor,
    grid: Optional[Sequence[float]] = None,
    conf_mode: str = "max_prob",
) -> Tuple[float, List[Tuple[float, float]]]:
    """Algorithm 2 step 1: pick lambda* maximizing validation accuracy over `grid`.

    `grid` defaults to `PAPER_LAMBDA_GRID`, the paper-reproduction grid used
    by this release.
    """
    if grid is None:
        grid = list(PAPER_LAMBDA_GRID)
    history: List[Tuple[float, float]] = []
    best_lam, best_acc = 0.0, -1.0
    for lam in grid:
        pred = simulate_ala(base_logits_val, P_val, lam, conf_mode)
        acc = float((pred == y_val).float().mean())
        history.append((float(lam), acc))
        if acc > best_acc:
            best_acc, best_lam = acc, float(lam)
    return best_lam, history


try:
    from transformers.generation.logits_process import LogitsProcessor
except ImportError:
    class LogitsProcessor:
        def __call__(self, input_ids, scores):
            raise NotImplementedError


class OLAAdaptiveProcessor(LogitsProcessor):
    """Algorithm 2 step 2: apply the Eq. 7 digit-only correction inside a real
    ``transformers.generate()`` loop, once, at the answer slot."""

    def __init__(
        self,
        lens,
        digit_ids: Sequence[int],
        lambda_val: float,
        captured_h_by_layer: Dict[int, torch.Tensor],
        eos_ids: Sequence[int],
        conf_mode: str = "max_prob",
        answer_slot_detector: Optional[Callable[[object, torch.Tensor], bool]] = None,
        apply_once: bool = True,
        require_answer_slot: bool = False,
    ):
        if not isinstance(captured_h_by_layer, dict):
            raise TypeError("captured_h_by_layer must be the dict shared with hooks")
        if require_answer_slot and answer_slot_detector is None:
            raise ValueError("answer_slot_detector is required when require_answer_slot=True")
        self.lens = lens
        self.digit_ids = [int(t) for t in digit_ids]
        self.lambda_val = float(lambda_val)
        self.captured_h_by_layer = captured_h_by_layer
        self.eos_ids = set(int(e) for e in eos_ids)
        self.conf_mode = conf_mode
        self.answer_slot_detector = answer_slot_detector
        self.apply_once = bool(apply_once)
        self.require_answer_slot = bool(require_answer_slot)

        overlap = set(self.digit_ids) & self.eos_ids
        if overlap:
            raise ValueError(f"digit/EOS overlap would corrupt the invariant: {overlap}")

        self._P: Optional[torch.Tensor] = None
        self._P_signature = None
        self._has_applied = False
        self._sample_id = None

    def begin_sample(self, sample_id: object = None) -> None:
        self.reset()
        self._sample_id = sample_id

    def reset(self) -> None:
        self._P = None
        self._P_signature = None
        self._has_applied = False

    @property
    def has_applied(self) -> bool:
        return self._has_applied

    def _is_answer_slot(self, input_ids, scores: torch.Tensor) -> bool:
        if self.answer_slot_detector is None:
            return True
        return bool(self.answer_slot_detector(input_ids, scores))

    def _captured_signature(self):
        captured = self.captured_h_by_layer
        if not captured or any(l not in captured for l in self.lens.layers):
            return None
        signature = []
        for layer_id in self.lens.layers:
            h = captured[layer_id]
            if isinstance(h, torch.Tensor):
                data_ptr = h.data_ptr() if h.numel() else 0
                version = getattr(h, "_version", 0)
                signature.append(
                    (
                        int(layer_id),
                        id(h),
                        int(data_ptr),
                        tuple(h.shape),
                        str(h.dtype),
                        str(h.device),
                        int(version),
                    )
                )
            else:
                signature.append((int(layer_id), id(h)))
        return tuple(signature)

    def _ensure_P(self, device) -> bool:
        signature = self._captured_signature()
        if signature is None:
            return False
        if self._P is not None and self._P_signature == signature:
            return True
        self._P = None
        self._P_signature = signature
        self._has_applied = False
        captured = self.captured_h_by_layer
        h_by_layer = {
            l: captured[l].to(device=device, dtype=torch.float32) for l in self.lens.layers
        }
        with torch.no_grad():
            self._P = self.lens.predict_probs(h_by_layer)
        return True

    def __call__(self, input_ids, scores):
        if self.lambda_val == 0.0:
            return scores
        if not self._is_answer_slot(input_ids, scores):
            return scores
        if not self._ensure_P(scores.device):
            return scores
        if self.apply_once and self._has_applied:
            return scores
        idx = torch.tensor(self.digit_ids, device=scores.device, dtype=torch.long)
        digit_logits = scores.index_select(dim=1, index=idx)
        adjusted_digits = correct_digit_logits(
            digit_logits, self._P, self.lambda_val, self.conf_mode
        )
        out = scores.clone()
        out[:, idx] = adjusted_digits
        self._has_applied = True
        return out


__all__ = [
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
]
