![OLA method overview](assets/main.png)

# OLA — Inference-Time Ordinal Lens Alignment (minimal core)

> Code release for *Latent Ordinal Evidence, Misaligned Outputs: Inference-Time Ordinal Lens Alignment for Multimodal LLMs* (anonymous, under review).

This is a **minimal, self-contained implementation of the OLA method core**. It exposes the paper-facing lens-training and inference-time alignment primitives without model downloads, datasets, or checkpoints. The data pipelines, backbone adapters, and the heavy multi-backbone experimental harness are intentionally **not** part of this MVP.

A synthetic end-to-end demo (`examples/quickstart_synthetic.py`) runs the whole pipeline — Eq. 4 lens, Eq. 6 fusion, Algorithm 1 training, Eq. 7 correction, Algorithm 2 `lambda*` search — in a few seconds with no model or dataset. The demo uses controlled toy tensors with planted ordinal evidence and biased native digit logits; it is a smoke test of the method pipeline, not a reproduction of the paper benchmark results.

## What OLA does

OLA is a **frozen-backbone, inference-time** procedure. The vision encoder, language-model backbone, unembedding `W_U`, and the digit-token row slice `W_S` are all frozen; the only learned parameters are lightweight per-layer lens weights and a `K`-dim fusion vector.

1. **`W_S`-anchored ordinal lens** (Eq. 4) — each of `K` mid-to-deep decoder layers maps its last-token hidden state to digit-token scores through the **frozen** `W_S`:

   `z_hat_l = W_S (A_l h_tilde_l + b_l)`,  `A_l = I + U_l V_l^T`,  `h_tilde_l = (h_l - mu_l) / sigma_l`

2. **Multi-layer fusion to `P^OLA`** (Eq. 6) — a learned softmax over layers:

   `a = softmax(alpha)`,  `z^OLA = sum_l a_l z_hat_l`,  `P^OLA = softmax(z^OLA)`

3. **Inference-time, digit-only logit correction** (Eq. 7) — applied to digit logits only; EOS and all non-digit logits are left untouched:

   `L'(c) = L_na(c) - lambda_star * omega(P^OLA) * (Q(c) - P^OLA(c))`,  `c in S_digit`

   where `Q = softmax(L_na[S_digit])`, `omega = max_c P^OLA(c)` is a confidence gate, and `lambda_star` is selected on the validation split.

`W_S` is verified by a SHA-256 hash before/after training so it provably never changes.

## Install

```bash
pip install -e .              # installs the `ola` package (torch + numpy)
pip install -e ".[hf]"        # + transformers, for real transformers.generate() integration
pip install -e ".[dev]"       # + pytest, to run the invariant tests
```

## Quickstart

```bash
python examples/quickstart_synthetic.py     # full pipeline on synthetic tensors, ~seconds
pytest -q                                    # method invariants (lambda=0 no-op, frozen W_S, sign, ...)
```

## Layout

```
ola/
  lens.py        # Eq.4 W_S-anchored lens, Eq.6 multi-layer fusion, frozen-W_S (SHA-256) checks
  train.py       # records, forward hooks, hidden-state cache, Stage A/B training, P^OLA cache, OLAConfig
  alignment.py   # tokenizer S_digit / S_eos ids, Eq.7 correction, lambda* search, online processor
  metrics.py     # accuracy, MAE, SRCC
examples/
  quickstart_synthetic.py   # end-to-end demo, no model download
tests/
  test_invariants.py        # fast correctness invariants
```

### Paper-to-code map

| Paper component | MVP implementation |
| --- | --- |
| Method hyperparameters (Appendix C) | `ola/train.py::OLAConfig` |
| Dataset records `(x, P, y)` | `ola/train.py::prepare_ordinal_records` |
| Real MLLM forward hooks | `ola/train.py::register_last_token_hooks` |
| Last-token hidden-state cache | `ola/train.py::collect_last_token_hidden_cache` |
| Eq. 4 `W_S`-anchored ordinal lens | `ola/lens.py::LowRankOrdinalLens` |
| Eq. 6 multi-layer fusion into `P^OLA` | `ola/lens.py::MultiLayerOrdinalLens.predict_probs` |
| Algorithm 1 Stage A / Stage B | `ola/train.py::train_ola_lens` |
| Algorithm 1 `P^OLA` cache | `ola/train.py::cache_p_ola` |
| Algorithm 2 `lambda*` validation sweep | `ola/alignment.py::search_lambda` |
| Algorithm 2 grid `Lambda` | `ola/alignment.py::PAPER_LAMBDA_GRID` |
| Tokenizer-specific `S_digit`, `S_eos` | `ola/alignment.py::resolve_digit_and_eos_ids` |
| Eq. 7 digit-only correction | `ola/alignment.py::correct_digit_logits` |
| Algorithm 2 online correction primitive | `ola/alignment.py::OLAAdaptiveProcessor` |

`OLAConfig` centralizes the paper-facing knobs (`layers`, `rank`, Stage A/B
training settings, `lambda_grid`, confidence gate, and online apply-once policy).
Use `OLAConfig.build_lens(...)`, `OLAConfig.train_config(...)`,
`OLAConfig.lambda_values()`, and `OLAConfig.processor_kwargs()` to pass the same
method configuration through both phases. The defaults follow the submitted experiment configuration:
`layers = {16, 18, 20, 22, 24, 26}` (`K = 6`), `rank = 64`, Stage A AdamW
`lr = 1e-4` (weight decay `1e-2`) for `<=100` epochs, Stage B AdamW `lr = 1e-2`
(no weight decay) for `<=30` epochs.

### The `lambda` grid

Algorithm 2 denotes the validation search set abstractly as `Lambda`. This
release uses the paper-reproduction grid
`PAPER_LAMBDA_GRID = (0, 0.5, 1, 2, 3, 5, 8, 10, 15, 20)` as the default for
both `search_lambda(...)` and `OLAConfig`. `default_grid(n, hi)` is also
provided for an optional finer uniform sweep; pass its output as the `grid`
argument of `search_lambda` if you want one.

## Using OLA with a real MLLM

The MVP keeps backbone wiring out of scope. `OLAAdaptiveProcessor` implements the Eq. 7 digit-only correction for a real `transformers.generate()` loop and supports prompt-specific answer-slot control via `answer_slot_detector`:

1. Use `prepare_ordinal_records(...)` to convert dataset-specific samples into `(image, designed prompt, ordinal label)` records.
2. Resolve the model-specific decoder layer modules, then call `register_last_token_hooks(...)` or `collect_last_token_hidden_cache(...)` to cache each layer's last-token hidden state from a frozen forward pass.
3. Build a `MultiLayerOrdinalLens` whose `W_S` is the LM head's digit-token rows, and train (`train_ola_lens`) or load Phase-1 lens weights.
4. Use `resolve_digit_and_eos_ids(tokenizer, n_classes, ...)` to obtain tokenizer-specific `S_digit` and `S_eos`.
5. Construct `OLAAdaptiveProcessor(lens, digit_ids, lambda_star, captured_h_by_layer=that_dict, eos_ids=..., answer_slot_detector=...)` and pass it through `LogitsProcessorList`.
6. For each sample, clear the shared captured-hidden dictionary, call `processor.begin_sample(sample_id)`, then run the prefill forward pass so hooks populate `H[l]`.
7. During generation, the processor computes `P^OLA` from the captured hidden states and applies Eq. 7 to the digit columns only at the answer slot. With `apply_once=True`, later generation steps are left unchanged until the next `begin_sample(...)` call.

## Citation

```bibtex
@inproceedings{anonymous2026ola,
  title  = {Latent Ordinal Evidence, Misaligned Outputs: Inference-Time Ordinal Lens Alignment for Multimodal LLMs},
  author = {Anonymous},
  year   = {2026}
}
```

Licensed under Apache-2.0. See `LICENSE`.
