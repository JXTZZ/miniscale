# Checkpoint and resume contract

MiniScale treats "resume" as continuation of the same training experiment, not
as loading weights for a new experiment. All five production stages use format v2
and stage-specific versioned resume identities.

## What strict resume verifies

Before model or optimizer state is mutated, the production training entry point checks:

- SHA-256 identities of the training data and optional dedicated validation data;
- tokenizer class, vocabulary, special-token IDs, and local tokenizer files;
- complete model configuration;
- optimizer hyperparameters, LR schedule, clipping, batch, accumulation, context,
  worker count, seed, validation split, and shuffle-buffer settings;
- pretraining implementation and signature versions.

SFT additionally verifies the parent initialization checkpoint, assistant-turn
example format, reasoning target mode, structured mask and truncation versions,
exact-dedup policy, global indexed data order, and minimum retained context.

DPO additionally verifies the parent SFT checkpoint, frozen reference identity,
shared-prompt pair format, target mode, pair-aware truncation, beta, prompt-hash
split, exact-dedup policy, and global indexed pair order. `reference.pt` is an
immutable model-only snapshot stored once in the DPO output directory; exact
resume verifies its SHA-256 before restoring optimizer state.

Paths are recorded for provenance, but compatibility uses content identities, so
moving an unchanged dataset or tokenizer does not invalidate a checkpoint.
Logging, W&B, and checkpoint-retention settings can be changed on resume.
Keep any settings that affect the stage's resume signature, including the SFT
generation suite and cadence used for early stopping.

Every run writes a stage manifest such as `pretrain_run.json`, with its resolved
recipe and input identities. Full checkpoints include the same identity plus model,
optimizer, scheduler, progress counters, and Python/NumPy/PyTorch/CUDA RNG state.

Emergency checkpoints are written before an optimizer update or after the
whole step completes. An interrupt during an update or its validation falls
back to the latest checkpoint successfully written by the run, when one exists.

SFT best-loss checkpoints include any generation and early-stopping state from
the same step. When an older best-loss checkpoint is missing that state, resume
replays that step's deterministic generation evaluation. A checkpoint that
already reached early stopping can initialize a new run via `--checkpoint`.

On resume, the local W&B upload queue drops events newer than the checkpoint.
Metrics already uploaded to W&B remain in the remote run history.

Agent RL's Python API fingerprints the complete in-memory task definition.
Older in-memory Agent checkpoints can initialize a new run via `--checkpoint`.
JSONL checkpoints continue to use the source-file identity.

LoRA checkpoints save the frozen base, the adapter tensors, and the adapter
settings. Repeat the same LoRA arguments for SFT resume. Inference and stage
initialization load the merged weights. The RL resume signature also records
enabled CISPO, batched sampling, and KV cache options because they can affect
the training trajectory.

## Checkpoints from the previous layout

The checkpoint reader accepts the old serialized `miniscale.config.MiniScaleConfig`
class for inference, stage initialization, and exact resume. New files store
the same configuration as a dictionary. The bundled tokenizer's fingerprint
uses the original two JSON files. A format-v2 checkpoint can resume using its
original training inputs and options. Keep the existing `reference.pt` next to
the training outputs when resuming DPO, GRPO, or Agent RL.

## Pretraining checkpoints before format v2

Checkpoints created before format v2 do not contain enough information to prove
that data, tokenizer, and model semantics are unchanged. They are rejected by
default. If you have manually verified those inputs, perform a one-time migration:

```bash
uv run python trainer/train_pretrain.py \
  --steps 10000 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --sequence-length 512 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 200 \
  --precision fp32 \
  --resume artifacts/pretrain/checkpoints/step_00000500.pt \
  --allow-legacy-resume \
  --output artifacts/pretrain
```

MiniScale still compares every legacy signature field that exists and emits a
warning about the fields it cannot prove. The next periodic, best, emergency, or
final checkpoint is saved as v2; subsequent resumes no longer need the flag.
Legacy AdamW checkpoints that used one parameter group are migrated by parameter
order into the current matrix-weight decay and norm/embedding no-decay groups.
This is intentionally warning-gated because the regularization policy changes at
the migration boundary.

`pretrain.pt` files from the smoke path contain model weights only. They are valid
for inference or stage hand-off but cannot be migrated into resumable optimizer
checkpoints.

## Output safety

Starting a new production run in a directory that already contains stage
metrics or checkpoints is rejected. Choose a new `--output`, or use `--resume`
with the matching checkpoint. This prevents accidental metric concatenation and
checkpoint overwrite.

Dedicated validation data must not be byte-for-byte identical to the training
data. Near-duplicate and benchmark-contamination checks remain a separate data
governance responsibility.

For SFT, `--checkpoint` starts a new run from model weights and creates a new
optimizer/scheduler. `--resume` requires a full SFT checkpoint and restores the
same run exactly. A legacy pretraining `best.pt` is therefore a valid SFT
`--checkpoint`, but it is never an SFT `--resume` checkpoint.

For DPO, `--checkpoint` must be an SFT checkpoint and creates both the initial
policy and frozen `reference.pt`. `--resume` requires a full DPO checkpoint plus
the unchanged `reference.pt` in the same output directory. Final `dpo.pt` keeps
top-level model/config fields for inference and GRPO hand-off.
