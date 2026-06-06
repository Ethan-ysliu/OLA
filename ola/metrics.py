from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_np(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy(pred: ArrayLike, true: ArrayLike) -> float:
    p, t = _to_np(pred).reshape(-1), _to_np(true).reshape(-1)
    return float((p == t).mean())


def mae(pred: ArrayLike, true: ArrayLike) -> float:
    p, t = _to_np(pred).astype(float).reshape(-1), _to_np(true).astype(float).reshape(-1)
    return float(np.abs(p - t).mean())


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and sorted_a[end] == sorted_a[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def srcc(scores: ArrayLike, true: ArrayLike) -> float:
    s, t = _to_np(scores).astype(float).reshape(-1), _to_np(true).astype(float).reshape(-1)
    if len(s) < 2:
        return 0.0
    rs, rt = _rankdata(s), _rankdata(t)
    rs, rt = rs - rs.mean(), rt - rt.mean()
    denom = np.sqrt((rs**2).sum() * (rt**2).sum())
    return 0.0 if denom == 0 else float((rs * rt).sum() / denom)


__all__ = ["accuracy", "mae", "srcc"]
