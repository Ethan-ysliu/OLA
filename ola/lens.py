from __future__ import annotations

import hashlib
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


class LowRankOrdinalLens(nn.Module):

    def __init__(self, dim: int, rank: int, init_scale: float = 0.02):
        super().__init__()
        if not 0 < rank <= dim:
            raise ValueError(f"need 0 < rank <= dim; got rank={rank}, dim={dim}")
        self.dim = dim
        self.rank = rank
        self.init_scale = float(init_scale)
        self.U = nn.Parameter(torch.empty(dim, rank))
        self.V = nn.Parameter(torch.empty(dim, rank))
        self.b = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        U = torch.randn(self.dim, self.rank, generator=gen) * self.init_scale
        with torch.no_grad():
            self.U.copy_(U.to(device=self.U.device, dtype=self.U.dtype))
            self.V.zero_()
            self.b.zero_()

    def correct(self, h_norm: torch.Tensor) -> torch.Tensor:

        delta = (h_norm @ self.V) @ self.U.t()
        return h_norm + delta + self.b

    def forward(self, h_norm: torch.Tensor, W_S: torch.Tensor) -> torch.Tensor:
        return self.correct(h_norm) @ W_S.t()


class MultiLayerOrdinalLens(nn.Module):

    def __init__(
        self,
        layers: Sequence[int],
        dim: int,
        n_classes: int,
        rank: int,
        W_S: torch.Tensor,
        init_scale: float = 0.02,
    ):
        super().__init__()
        if W_S.shape != (n_classes, dim):
            raise ValueError(f"W_S shape {tuple(W_S.shape)} must be ({n_classes}, {dim})")

        self.layers: List[int] = sorted(int(layer) for layer in layers)
        self.dim = int(dim)
        self.n_classes = int(n_classes)
        self.rank = int(rank)
        self.init_scale = float(init_scale)

        if not self.layers:
            raise ValueError("`layers` must be non-empty")

        self.lenses = nn.ModuleList(
            [
                LowRankOrdinalLens(dim=dim, rank=rank, init_scale=self.init_scale)
                for _ in self.layers
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(len(self.layers)))

        W_S_frozen = W_S.detach().clone().to(torch.float32)
        W_S_frozen.requires_grad_(False)
        self.register_buffer("W_S", W_S_frozen, persistent=True)
        self._W_S_hash = sha256_tensor(self.W_S)

        self.register_buffer("norm_mu", torch.zeros(len(self.layers), dim), persistent=True)
        self.register_buffer("norm_sigma", torch.ones(len(self.layers), dim), persistent=True)
        self._norm_set: List[bool] = [False] * len(self.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _layer_idx(self, layer_id: int) -> int:
        try:
            return self.layers.index(int(layer_id))
        except ValueError as exc:
            raise KeyError(f"layer {layer_id} not in {self.layers}") from exc

    def set_lenses_trainable(self, flag: bool) -> None:
        for lens in self.lenses:
            for p in lens.parameters():
                p.requires_grad_(flag)

    def set_alpha_trainable(self, flag: bool) -> None:
        self.alpha.requires_grad_(flag)

    def assert_W_S_frozen(self) -> None:
        if self.W_S.requires_grad:
            raise RuntimeError("W_S must stay frozen")
        cur = sha256_tensor(self.W_S)
        if cur != self._W_S_hash:
            raise RuntimeError(
                f"W_S hash drift: expected {self._W_S_hash[:8]}, got {cur[:8]}"
            )

    def set_norm_stats(
        self,
        layer_id: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        sigma_floor: float = 1e-3,
    ) -> None:
        idx = self._layer_idx(layer_id)
        if mu.shape != (self.dim,) or sigma.shape != (self.dim,):
            raise ValueError(
                f"mu/sigma must be ({self.dim},); got "
                f"mu={tuple(mu.shape)} sigma={tuple(sigma.shape)}"
            )
        with torch.no_grad():
            self.norm_mu[idx].copy_(mu.to(self.norm_mu.dtype))
            self.norm_sigma[idx].copy_(
                sigma.to(self.norm_sigma.dtype).clamp(min=sigma_floor)
            )
        self._norm_set[idx] = True

    def set_norm_stats_from(
        self, h_by_layer: Mapping[int, torch.Tensor], sigma_floor: float = 1e-3
    ) -> None:
        for layer_id, h in h_by_layer.items():
            self.set_norm_stats(
                layer_id, h.float().mean(0), h.float().std(0), sigma_floor
            )

    def normalize(self, layer_id: int, h: torch.Tensor) -> torch.Tensor:
        idx = self._layer_idx(layer_id)
        if not self._norm_set[idx]:
            raise RuntimeError(f"norm stats for layer {layer_id} not set")
        return (h.float() - self.norm_mu[idx]) / self.norm_sigma[idx]

    def single_layer_logits(self, layer_id: int, h: torch.Tensor) -> torch.Tensor:
        idx = self._layer_idx(layer_id)
        return self.lenses[idx](self.normalize(layer_id, h), self.W_S)

    def per_layer_logits(self, h_by_layer: Mapping[int, torch.Tensor]) -> torch.Tensor:
        zs = []
        for layer_id in self.layers:
            if layer_id not in h_by_layer:
                raise KeyError(f"missing hidden for layer {layer_id}")
            h = h_by_layer[layer_id]
            if h.dim() != 2 or h.shape[1] != self.dim:
                raise ValueError(
                    f"layer {layer_id} expected [B, {self.dim}], got {tuple(h.shape)}"
                )
            zs.append(self.single_layer_logits(layer_id, h))
        return torch.stack(zs, dim=1)

    def fusion_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha, dim=0)

    def predict_logits(self, h_by_layer: Mapping[int, torch.Tensor]) -> torch.Tensor:
        z = self.per_layer_logits(h_by_layer)
        return torch.einsum("bkc,k->bc", z, self.fusion_weights())

    def predict_probs(self, h_by_layer: Mapping[int, torch.Tensor]) -> torch.Tensor:
        return F.softmax(self.predict_logits(h_by_layer), dim=-1)

    def forward(
        self, h_by_layer: Mapping[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_per = self.per_layer_logits(h_by_layer)
        a = self.fusion_weights()
        z = torch.einsum("bkc,k->bc", z_per, a)
        return z, z_per, a

    def to_state_payload(self, extra: Optional[dict] = None) -> dict:
        state = {k: v for k, v in self.state_dict().items() if k != "W_S"}
        return {
            "model_name": "OLA::MultiLayerOrdinalLens",
            "layers": list(self.layers),
            "dim": self.dim,
            "n_classes": self.n_classes,
            "rank": self.rank,
            "init_scale": self.init_scale,
            "state_dict": state,
            "W_S_hash": self._W_S_hash,
            "norm_set": list(self._norm_set),
            "extra": extra or {},
        }

    @classmethod
    def from_state_payload(
        cls, payload: dict, W_S: torch.Tensor, strict_hash: bool = True
    ) -> "MultiLayerOrdinalLens":
        if payload.get("model_name") != "OLA::MultiLayerOrdinalLens":
            raise ValueError(f"bad payload model_name={payload.get('model_name')!r}")
        module = cls(
            layers=payload["layers"],
            dim=payload["dim"],
            n_classes=payload["n_classes"],
            rank=payload["rank"],
            W_S=W_S,
            init_scale=payload.get("init_scale", 0.02),
        )
        cur = sha256_tensor(module.W_S)
        if cur != payload["W_S_hash"]:
            msg = f"W_S hash drift: payload={payload['W_S_hash'][:8]} runtime={cur[:8]}"
            if strict_hash:
                raise RuntimeError(msg)
            print(f"  [WARN] {msg}")

        state = dict(payload["state_dict"])
        state.pop("W_S", None)
        expected_keys = set(module.state_dict().keys()) - {"W_S"}
        found_keys = set(state.keys())
        missing = sorted(expected_keys - found_keys)
        unexpected = sorted(found_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "invalid OLA lens checkpoint keys: "
                f"missing={missing}, unexpected={unexpected}"
            )

        full_state = dict(state)
        full_state["W_S"] = module.W_S
        module.load_state_dict(full_state, strict=True)
        module._norm_set = list(payload.get("norm_set", [True] * module.n_layers))
        module.W_S.requires_grad_(False)
        module.assert_W_S_frozen()
        return module


__all__ = ["LowRankOrdinalLens", "MultiLayerOrdinalLens", "sha256_tensor"]
