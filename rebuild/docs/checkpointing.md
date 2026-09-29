# Checkpoint and resume contract

MiniScale treats "resume" as continuation of the same training experiment, not
as loading weights for a new experiment. Production pretraining, SFT, and DPO
checkpoints therefore use checkpoint format v2 and stage-specific versioned
resume identities.

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
Logging, W&B, generation, checkpoint-retention, and output cadence may be changed
when resuming because they do not alter optimizer updates.

Every run also writes `pretrain_run.json`, `sft_run.json`, or `dpo_run.json`, a human-readable resolved recipe and
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

## Implementation v3 boundary

All five production stages now record implementation version 3. RoPE preserves
BF16 Q/K dtype; generation probabilities use FP32; packed pretraining uses the
native causal attention path. Multi-worker pretraining advances the shuffle seed
between epochs, and dedicated validation no longer excludes a hash slice from
the training file. RL optimization keeps dropout disabled, matching rollout and
old-policy scoring.

These changes can alter a training trajectory. Existing weights remain loadable
for inference, HF export and stage handoff, but an implementation-v2 checkpoint
cannot silently resume under v3. Finish an old run with its original code, or
start a new experiment from its weights. `--allow-legacy-resume` remains limited
to the documented legacy signature migration; it does not bypass a v2/v3
implementation mismatch in a strict checkpoint.

Emergency saving tracks optimizer update boundaries. Before optimizer mutation,
partial gradients are discarded and RNG is rewound to the beginning of the
unfinished update; committed data counters still identify the previous update.
Once mutation starts, interruption can leave AdamW, an RL policy epoch, or
validation/early-stopping bookkeeping only partially applied. No emergency
checkpoint is published in that case: use the last complete periodic/best checkpoint. This avoids cloning all model and AdamW
state every step. SIGKILL and power loss still require a previously saved file.

Checkpoints deserialize on CPU before state restoration, so inference does not
load optimizer tensors into GPU memory. Loading weights preserves the caller's
CPU RNG. SFT best-loss snapshots include the same step's generation and early
stopping counters. W&B removes queued events later than the resumed step before
upload; history already uploaded to W&B cannot be rolled back by local code.

Run manifests also record Python/PyTorch/Transformers/NumPy versions, CUDA build,
device, thread count, TF32 and deterministic-algorithm settings. Fixed seeds and
input identities support replay in the same runtime; they do not promise bitwise
identity across hardware or library versions.
