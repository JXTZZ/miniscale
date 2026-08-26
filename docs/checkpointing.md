# Checkpoint and resume contract

MiniScale treats "resume" as continuation of the same training experiment, not
as loading weights for a new experiment. New pretraining checkpoints therefore
use checkpoint format v2 and a versioned resume identity.

## What strict resume verifies

Before model or optimizer state is mutated, the pretraining entry point checks:

- SHA-256 identities of the training data and optional dedicated validation data;
- tokenizer class, vocabulary, special-token IDs, and local tokenizer files;
- complete model configuration;
- optimizer hyperparameters, LR schedule, clipping, batch, accumulation, context,
  worker count, seed, validation split, and shuffle-buffer settings;
- pretraining implementation and signature versions.

Paths are recorded for provenance, but compatibility uses content identities, so
moving an unchanged dataset or tokenizer does not invalidate a checkpoint.
Logging, W&B, generation, checkpoint-retention, and output cadence may be changed
when resuming because they do not alter optimizer updates.

Every run also writes `pretrain_run.json`, a human-readable resolved recipe and
input identity manifest. Full checkpoints contain the same identity plus model,
optimizer, scheduler, progress counters, and Python/NumPy/PyTorch/CUDA RNG state.

## Legacy checkpoint migration

Checkpoints created before format v2 do not contain enough information to prove
that data, tokenizer, and model semantics are unchanged. They are rejected by
default. If you have manually verified those inputs, perform a one-time migration:

```bash
uv run miniscale pretrain \
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

Starting a new production run in a directory that already contains pretraining
metrics or checkpoints is rejected. Choose a new `--output`, or use `--resume`
with the matching checkpoint. This prevents accidental metric concatenation and
checkpoint overwrite.

Dedicated validation data must not be byte-for-byte identical to the training
data. Near-duplicate and benchmark-contamination checks remain a separate data
governance responsibility.
