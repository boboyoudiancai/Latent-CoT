# LaRA research implementation on StarVLA

This branch publishes our StarVLA implementation to the project-owned
`boboyoudiancai/Latent-CoT` repository. It follows the upstream StarVLA history
at commit `985c223`, rather than replacing or merging the LaRA `main` branch.
The upstream license and attribution are retained.

## Snapshot scope

The November 7, 2026 snapshot contains the five modified tracked files and
30 new implementation, configuration, evaluation and test files from our
StarVLA working directory. This document is the only publication-specific
addition. The source files are copied without implementation changes.

Included research features:

- Future prediction with 16 query tokens and EMA-RMS residual targets.
- Spatial Forcing auxiliary supervision with a frozen VGGT teacher.
- Three-component latent language supervision: subtask, bounding box, reasoning.
- Two-stage thinking and language-only ablations.
- Single-stage action training with a separately loaded frozen 4B decoder and
  a trainable residual MLP latent-input projector.
- An opt-in `independent_queries` ablation: three independent input embeddings,
  one VLM pass and same-position output hidden states for decoder supervision.
- A QwenPI LIBERO-Plus parallel evaluator with training-contract checks.
- Gradient-accumulation, optimizer-group, scheduler and checkpoint/resume changes.

The original recursive thinking mode remains the default. The independent-query
configuration differs from the frozen-decoder recursive configuration only in
the forward mode and run identifier, apart from comments.

## Current training configuration

`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_frozen4b_independent_queries_single_stage.yaml`
uses three independent queries, a frozen 4B decoder, a trainable residual MLP
projector and `action_loss + 0.1 * decoder_loss`. Future and Spatial losses are
disabled. It specifies 40K steps, a 5K optimizer-step warmup, per-device batch 14
and gradient accumulation 1 (global batch 112 on eight GPUs).

The historical two-stage thinking Stage2 YAML separately specifies 100K steps.

## Validation at publication

Checks ran on CPU using the existing StarVLA environment and a locally available
Qwen processor. No training process or training source directory was changed.

- `git diff --check`: passed.
- The 60-case CPU regression run: 58 passed, one assertion failure and one error;
  no tests were skipped when the local processor was configured.
- Gradient accumulation was also checked in its required fresh process:
  two microbatches produced one optimizer step and one scheduler step (passed).

The two pre-existing failing checks are preserved in this snapshot:

1. `tests/test_future_dataset_contract.py::test_future_fallback_uses_first_view_only`
   builds a dataset with `__new__` and omits the now-required
   `thinking_annotations` attribute, raising `AttributeError`.
2. `tests/test_thinking_checkpoint_validation.py::test_thinking_stage_configs`
   expects a 40K Stage2 configuration, while the checked-in historical thinking
   Stage2 configuration now specifies 100K steps.

The tests and configuration were not silently changed to make the snapshot pass.
The CPU run is not a new full-size GPU training or LIBERO-Plus evaluation.

## Environment and excluded artifacts

Training configurations retain their original cluster-specific model, data and
checkpoint paths. Configure these paths and external VGGT/LIBERO dependencies
for another installation. Processor integration tests use
`QWENPI_THINKING_PROCESSOR` to select an existing local model directory.

The environment used for validation did not have pytest installed. Unittest
cases were discovered with Python's standard library; parameterless function
tests were invoked through `unittest.FunctionTestCase`. The accumulation test
must run in a fresh process because it configures Accelerate before import.

Checkpoints, model weights, datasets, environments, evaluation results, logs and
the nine local `.bak` files are not part of the publication. The upstream
StarVLA source tree and its existing small documentation assets are retained.
