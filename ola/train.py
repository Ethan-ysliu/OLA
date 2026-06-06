from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lens import MultiLayerOrdinalLens


def prepare_ordinal_records(
    raw_items: Sequence[object],
    prompt_builder: Callable[[object], str],
    image_getter: Callable[[object], object],
    label_getter: Callable[[object], int],
) -> List[dict]:
    records = []
    for item in raw_items:
        records.append(
            {
                "image": image_getter(item),
                "prompt": prompt_builder(item),
                "label": int(label_getter(item)),
                "raw": item,
            }
        )
    return records


def _hidden_tensor_from_module_output(module_output) -> torch.Tensor:
    if isinstance(module_output, torch.Tensor):
        hidden = module_output
    elif isinstance(module_output, (tuple, list)) and module_output:
        hidden = module_output[0]
    elif hasattr(module_output, "last_hidden_state"):
        hidden = module_output.last_hidden_state
    else:
        raise TypeError("decoder layer output does not contain hidden states")

    if hidden.dim() != 3:
        raise ValueError(f"expected hidden states [B, T, D], got {tuple(hidden.shape)}")
    return hidden


def extract_last_token_hidden(module_output, token_position: int = -1) -> torch.Tensor:
    hidden = _hidden_tensor_from_module_output(module_output)
    return hidden[:, token_position, :]


def register_last_token_hooks(
    layer_modules: Mapping[int, nn.Module],
    captured_h_by_layer: Dict[int, torch.Tensor],
    token_position: int = -1,
    detach: bool = True,
    to_cpu: bool = False,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for layer_id, module in layer_modules.items():
        lid = int(layer_id)

        def hook(_module, _inputs, output, layer_id=lid):
            h = extract_last_token_hidden(output, token_position=token_position)
            if detach:
                h = h.detach()
            h = h.to(dtype=torch.float32)
            if to_cpu:
                h = h.cpu()
            captured_h_by_layer[layer_id] = h

        handles.append(module.register_forward_hook(hook))
    return handles


def remove_hooks(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def collect_last_token_hidden_cache(
    forward_fn: Callable[[object], object],
    records: Sequence[object],
    layer_modules: Mapping[int, nn.Module],
    label_getter: Optional[Callable[[object], int]] = None,
    token_position: int = -1,
    to_cpu: bool = True,
) -> Tuple[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    per_layer_chunks: Dict[int, List[torch.Tensor]] = {
        int(layer_id): [] for layer_id in layer_modules
    }
    labels: List[int] = []
    captured: Dict[int, torch.Tensor] = {}
    handles = register_last_token_hooks(
        layer_modules,
        captured,
        token_position=token_position,
        detach=True,
        to_cpu=to_cpu,
    )
    try:
        for record in records:
            captured.clear()
            with torch.no_grad():
                forward_fn(record)
            missing = [layer_id for layer_id in layer_modules if layer_id not in captured]
            if missing:
                raise RuntimeError(f"forward hooks did not capture layers: {missing}")
            for layer_id in layer_modules:
                per_layer_chunks[int(layer_id)].append(captured[int(layer_id)])
            if label_getter is not None:
                labels.append(int(label_getter(record)))
    finally:
        remove_hooks(handles)

    H = {
        layer_id: torch.cat(chunks, dim=0)
        for layer_id, chunks in per_layer_chunks.items()
    }
    y = torch.tensor(labels, dtype=torch.long) if label_getter is not None else None
    return H, y


@dataclass(frozen=True)
class OLAConfig:
    """Paper-facing knobs for both OLA phases (Appendix C). Defaults follow
    the submitted experiment setup: layers L={16,...,26} (K=6), rank r=64,
    Stage A AdamW lr 1e-4 (weight decay 1e-2) up to 100 epochs,
    Stage B AdamW lr 1e-2 (no
    weight decay) up to 30 epochs, and the paper-reproduction lambda grid used
    for Algorithm 2."""

    layers: Tuple[int, ...] = (16, 18, 20, 22, 24, 26)
    rank: int = 64
    lens_init_scale: float = 0.02
    stage_a_epochs: int = 100
    stage_a_lr: float = 1e-4
    stage_a_weight_decay: float = 1e-2
    stage_b_epochs: int = 30
    stage_b_lr: float = 1e-2
    batch_size: int = 128
    grad_clip: float = 1.0
    sigma_floor: float = 1e-3
    base_seed: int = 42
    device: str = "cpu"
    # Algorithm 2 discrepancy-strength grid used by this release.
    lambda_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)
    conf_mode: str = "max_prob"
    apply_once: bool = True
    require_answer_slot: bool = True

    def train_config(self, verbose: bool = True) -> "TrainConfig":
        return TrainConfig(
            stage_a_epochs=self.stage_a_epochs,
            stage_a_lr=self.stage_a_lr,
            stage_a_weight_decay=self.stage_a_weight_decay,
            stage_b_epochs=self.stage_b_epochs,
            stage_b_lr=self.stage_b_lr,
            batch_size=self.batch_size,
            grad_clip=self.grad_clip,
            sigma_floor=self.sigma_floor,
            base_seed=self.base_seed,
            device=self.device,
            verbose=verbose,
        )

    def build_lens(
        self, W_S: torch.Tensor, dim: int, n_classes: int
    ) -> MultiLayerOrdinalLens:
        return MultiLayerOrdinalLens(
            layers=self.layers,
            dim=dim,
            n_classes=n_classes,
            rank=self.rank,
            W_S=W_S,
            init_scale=self.lens_init_scale,
        )

    def lambda_values(self) -> List[float]:
        return list(self.lambda_grid)

    def processor_kwargs(
        self, answer_slot_detector: Optional[Callable[[object, torch.Tensor], bool]] = None
    ) -> Dict[str, object]:
        if self.require_answer_slot and answer_slot_detector is None:
            raise ValueError(
                "answer_slot_detector is required by OLAConfig.require_answer_slot=True"
            )
        kwargs = {
            "conf_mode": self.conf_mode,
            "apply_once": self.apply_once,
            "require_answer_slot": self.require_answer_slot,
        }
        if answer_slot_detector is not None:
            kwargs["answer_slot_detector"] = answer_slot_detector
        return kwargs


@dataclass
class TrainConfig:
    stage_a_epochs: int = 100
    stage_a_lr: float = 1e-4
    stage_a_weight_decay: float = 1e-2
    stage_b_epochs: int = 30
    stage_b_lr: float = 1e-2
    batch_size: int = 128
    grad_clip: float = 1.0
    sigma_floor: float = 1e-3
    base_seed: int = 42
    device: str = "cpu"
    verbose: bool = True


@dataclass
class TrainLog:
    stage_a_val_acc: Dict[int, float] = field(default_factory=dict)
    stage_b_val_acc: float = 0.0
    fusion_weights: List[float] = field(default_factory=list)


def _iter_minibatches(n: int, bs: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, bs):
        yield perm[i : i + bs]


def _val_acc_single(
    lens: MultiLayerOrdinalLens, layer_id: int, h: torch.Tensor, y: torch.Tensor
) -> float:
    with torch.no_grad():
        pred = lens.single_layer_logits(layer_id, h).argmax(-1)
    return float((pred == y).float().mean())


def _val_acc_fused(
    lens: MultiLayerOrdinalLens, H: Mapping[int, torch.Tensor], y: torch.Tensor
) -> float:
    with torch.no_grad():
        pred = lens.predict_logits(H).argmax(-1)
    return float((pred == y).float().mean())


def train_ola_lens(
    lens: MultiLayerOrdinalLens,
    H_train: Mapping[int, torch.Tensor],
    y_train: torch.Tensor,
    H_val: Mapping[int, torch.Tensor],
    y_val: torch.Tensor,
    cfg: Optional[TrainConfig] = None,
) -> TrainLog:
    cfg = cfg or TrainConfig()
    dev = torch.device(cfg.device)
    lens.to(dev)
    H_train = {l: h.to(dev) for l, h in H_train.items()}
    H_val = {l: h.to(dev) for l, h in H_val.items()}
    y_train, y_val = y_train.to(dev), y_val.to(dev)
    log = TrainLog()

    lens.set_norm_stats_from(H_train, sigma_floor=cfg.sigma_floor)

    lens.assert_W_S_frozen()
    lens.set_alpha_trainable(False)
    n = y_train.shape[0]
    for layer_id in lens.layers:
        idx = lens._layer_idx(layer_id)
        this_lens = lens.lenses[idx]
        this_lens.reset_parameters(seed=cfg.base_seed + int(layer_id))
        for other_i, other in enumerate(lens.lenses):
            for p in other.parameters():
                p.requires_grad_(other_i == idx)

        opt = torch.optim.AdamW(
            this_lens.parameters(),
            lr=cfg.stage_a_lr,
            weight_decay=cfg.stage_a_weight_decay,
        )
        gen = torch.Generator().manual_seed(cfg.base_seed + int(layer_id))
        best_acc, best_state = -1.0, None
        for _ in range(cfg.stage_a_epochs):
            this_lens.train()
            for batch in _iter_minibatches(n, cfg.batch_size, gen):
                opt.zero_grad()
                z = lens.single_layer_logits(layer_id, H_train[layer_id][batch])
                loss = F.cross_entropy(z, y_train[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(this_lens.parameters(), cfg.grad_clip)
                opt.step()
            acc = _val_acc_single(lens, layer_id, H_val[layer_id], y_val)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(this_lens.state_dict())
        if best_state is not None:
            this_lens.load_state_dict(best_state)
        log.stage_a_val_acc[int(layer_id)] = best_acc
        if cfg.verbose:
            print(f"  [Stage A] layer {layer_id:>3}: best val acc = {best_acc:.4f}")
    lens.assert_W_S_frozen()

    lens.set_lenses_trainable(False)
    lens.set_alpha_trainable(True)
    opt = torch.optim.AdamW([lens.alpha], lr=cfg.stage_b_lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.base_seed)
    best_acc, best_alpha = -1.0, None
    for _ in range(cfg.stage_b_epochs):
        for batch in _iter_minibatches(n, cfg.batch_size, gen):
            opt.zero_grad()
            z = lens.predict_logits({l: H_train[l][batch] for l in lens.layers})
            loss = F.cross_entropy(z, y_train[batch])
            loss.backward()
            opt.step()
        acc = _val_acc_fused(lens, H_val, y_val)
        if acc > best_acc:
            best_acc, best_alpha = acc, lens.alpha.detach().clone()
    if best_alpha is not None:
        with torch.no_grad():
            lens.alpha.copy_(best_alpha)
    lens.assert_W_S_frozen()

    log.stage_b_val_acc = best_acc
    log.fusion_weights = lens.fusion_weights().detach().cpu().tolist()
    if cfg.verbose:
        print(f"  [Stage B] fused best val acc = {best_acc:.4f}")
        print(f"  [Stage B] fusion weights a = {[round(w, 3) for w in log.fusion_weights]}")
    return log


def cache_p_ola(
    lens: MultiLayerOrdinalLens,
    splits: Mapping[str, Mapping[int, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    was_training = lens.training
    lens.eval()
    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for split_name, h_by_layer in splits.items():
            out[split_name] = lens.predict_probs(h_by_layer)
    lens.train(was_training)
    return out


__all__ = [
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
]
