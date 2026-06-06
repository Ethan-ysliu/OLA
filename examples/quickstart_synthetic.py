"""End-to-end OLA smoke test on synthetic tensors (no model or dataset download).

Runs the whole pipeline in a few seconds and exercises every paper equation:

    Eq. 4   per-layer W_S-anchored lens          (MultiLayerOrdinalLens)
    Eq. 6   learned-softmax multi-layer fusion    (predict_probs -> P^OLA)
    Alg. 1  Stage A / Stage B training            (train_ola_lens)
    Eq. 7   inference-time digit-only correction  (correct_digit_logits)
    Alg. 2  validation lambda* selection          (search_lambda over PAPER_LAMBDA_GRID)

The synthetic setup plants an ordinal signal in the hidden states (so the lens
can recover the label) and fabricates "native" digit logits with a central-
tendency bias (so the backbone-style output is wrong). OLA should route the
latent evidence into the digit logits and fix the prediction. These numbers are
not paper benchmark results; they only verify that the MVP pipeline is wired
correctly under controlled toy conditions.

    python examples/quickstart_synthetic.py
"""

from __future__ import annotations

import torch

from ola import (
    OLAConfig,
    PAPER_LAMBDA_GRID,
    correct_digit_logits,
    mae,
    accuracy,
    search_lambda,
    sha256_tensor,
    srcc,
    train_ola_lens,
)

C, D = 5, 48                       # ordinal classes, hidden dim
LAYERS = (0, 1, 2, 3)              # toy "mid-to-deep" layer window (K=4)
SIGNAL = 6.0                       # planted ordinal signal-to-noise


def make_split(n: int, W_S: torch.Tensor, seed: int):
    """Hidden states whose class direction aligns with the frozen digit rows,
    plus native digit logits that are biased toward the central class."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (n,), generator=g)
    H = {}
    for li, layer in enumerate(LAYERS):
        strength = SIGNAL * (0.5 + 0.5 * li / max(len(LAYERS) - 1, 1))  # deeper = cleaner
        H[layer] = strength * W_S[y] + torch.randn(n, D, generator=g)
    mid = (C - 1) / 2.0
    bias = -2.0 * (torch.arange(C).float() - mid).abs()                # central-tendency peak
    L_native = bias.unsqueeze(0).repeat(n, 1) + 0.3 * torch.randn(n, C, generator=g)
    return H, y, L_native


def main() -> None:
    torch.manual_seed(0)

    # Frozen digit-token row submatrix W_S (unit-norm rows -> near-orthogonal).
    W_S = torch.randn(C, D)
    W_S = W_S / W_S.norm(dim=1, keepdim=True)
    W_S_hash_before = sha256_tensor(W_S)

    H_tr, y_tr, _ = make_split(512, W_S, seed=1)
    H_va, y_va, L_va = make_split(128, W_S, seed=2)
    H_te, y_te, L_te = make_split(256, W_S, seed=3)

    # Small/fast config (the paper defaults are layers={16..26}, rank 64, 100/30 epochs).
    cfg = OLAConfig(
        layers=LAYERS,
        rank=8,
        stage_a_epochs=30,
        stage_b_epochs=15,
        device="cpu",
        require_answer_slot=False,
    )
    lens = cfg.build_lens(W_S=W_S, dim=D, n_classes=C)

    print("== Algorithm 1: lens training ==")
    log = train_ola_lens(lens, H_tr, y_tr, H_va, y_va, cfg.train_config(verbose=True))
    print(f"  fusion weights a = {[round(w, 3) for w in log.fusion_weights]}")

    # W_S must be byte-identical before and after training.
    assert sha256_tensor(lens.W_S) == W_S_hash_before, "W_S drifted during training!"
    print(f"  W_S frozen check OK (sha256 {W_S_hash_before[:8]} unchanged)\n")

    # Eq. 6: fused P^OLA on val/test.
    with torch.no_grad():
        P_va = lens.predict_probs(H_va)
        P_te = lens.predict_probs(H_te)

    lens_pred = P_te.argmax(-1)
    native_pred = L_te.argmax(-1)
    print("== Diagnostic gap (native digit output vs lens-recoverable evidence) ==")
    print(f"  native digit-token  : ACC {accuracy(native_pred, y_te):.3f}  "
          f"MAE {mae(native_pred, y_te):.3f}")
    print(f"  lens P^OLA (Eq.6)   : ACC {accuracy(lens_pred, y_te):.3f}  "
          f"MAE {mae(lens_pred, y_te):.3f}  "
          f"SRCC {srcc(P_te.argmax(-1), y_te):.3f}\n")

    # Algorithm 2 step 1: select lambda* on validation over the release grid.
    print("== Algorithm 2: lambda* selection over PAPER_LAMBDA_GRID ==")
    print(f"  grid = {list(PAPER_LAMBDA_GRID)}")
    lam_star, history = search_lambda(L_va, P_va, y_va)
    print("  (lambda, val_acc): " + ", ".join(f"{l:g}:{a:.3f}" for l, a in history))
    print(f"  selected lambda* = {lam_star:g}\n")

    # Eq. 7: digit-only correction on test with the selected lambda*.
    corrected_pred = correct_digit_logits(L_te, P_te, lam_star).argmax(-1)
    print("== Eq. 7: inference-time correction on test ==")
    print(f"  before (native)     : ACC {accuracy(native_pred, y_te):.3f}  "
          f"MAE {mae(native_pred, y_te):.3f}")
    print(f"  after  (OLA-Online) : ACC {accuracy(corrected_pred, y_te):.3f}  "
          f"MAE {mae(corrected_pred, y_te):.3f}")

    assert accuracy(corrected_pred, y_te) > accuracy(native_pred, y_te), \
        "correction should improve over the biased native output"
    print("\nOLA routed the latent ordinal evidence into the digit-token output. Done.")


if __name__ == "__main__":
    main()
