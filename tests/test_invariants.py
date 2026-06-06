"""Fast invariant tests for the OLA method core (no model/dataset download).

Run with:  pytest -q   (or: python tests/test_invariants.py)
"""

from __future__ import annotations

import torch

from ola import (
    MultiLayerOrdinalLens,
    OLAConfig,
    PAPER_LAMBDA_GRID,
    correct_digit_logits,
    resolve_digit_and_eos_ids,
    sha256_tensor,
)


def _make_lens(C=5, D=16, layers=(0, 1, 2)):
    torch.manual_seed(0)
    W_S = torch.randn(C, D)
    lens = MultiLayerOrdinalLens(layers=layers, dim=D, n_classes=C, rank=4, W_S=W_S)
    for layer in layers:
        lens.set_norm_stats(layer, torch.zeros(D), torch.ones(D))
    return lens, W_S


def test_paper_lambda_grid_is_exact():
    # The default search grid must equal the paper-reproduction grid used by
    # this release for Algorithm 2.
    assert list(PAPER_LAMBDA_GRID) == [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    assert list(OLAConfig().lambda_values()) == list(PAPER_LAMBDA_GRID)


def test_lambda_zero_is_noop():
    logits = torch.randn(8, 5)
    P = torch.softmax(torch.randn(8, 5), dim=-1)
    out = correct_digit_logits(logits, P, lam=0.0)
    assert torch.equal(out, logits)


def test_correction_sign_is_signaware():
    # When Q over-favours class c relative to P^OLA, Eq.7 lowers that logit.
    logits = torch.tensor([[5.0, 0.0, 0.0]])      # Q peaks on class 0
    P = torch.tensor([[0.0, 0.0, 1.0]])           # P^OLA wants class 2
    out = correct_digit_logits(logits, P, lam=10.0)
    assert out[0, 0] < logits[0, 0]               # class 0 pushed down
    assert out[0, 2] > logits[0, 2]               # class 2 pushed up


def test_identity_init_gives_plain_WS_readout():
    # At init A_l = I + U V^T with V=0 => A_l = I, so z_hat = W_S @ h_norm.
    lens, W_S = _make_lens()
    h = torch.randn(4, 16)
    z = lens.single_layer_logits(0, h)            # norm stats are (0, 1) => h_norm = h
    assert torch.allclose(z, h @ W_S.t(), atol=1e-5)


def test_fusion_weights_are_a_simplex():
    lens, _ = _make_lens()
    a = lens.fusion_weights()
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-6)
    assert (a >= 0).all()


def test_W_S_frozen_through_forward():
    lens, _ = _make_lens()
    before = sha256_tensor(lens.W_S)
    H = {layer: torch.randn(4, 16) for layer in lens.layers}
    _ = lens.predict_probs(H)
    lens.assert_W_S_frozen()
    assert sha256_tensor(lens.W_S) == before


def test_digit_eos_overlap_is_rejected():
    class _Tok:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [int(text)]

    try:
        resolve_digit_and_eos_ids(_Tok(), n_classes=5)  # digit id 2 collides with EOS
    except ValueError:
        return
    raise AssertionError("expected digit/EOS overlap to raise ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all invariants passed")
