# Production pretraining contract

本文记录当前代码实际保证的预训练语义。正式路径是 CLI `miniscale pretrain` 调用
`src/miniscale/training/pretrain.py::run_pretrain_jsonl`；`trainer/train_pretrain.py` 和
`run_pretrain` 是独立的内存 smoke 路径，不代表正式 recipe。

## Objective and sequence construction

模型执行 decoder-only causal language modeling。每篇 JSONL 文档读取 `text` 字段并编码为
`BOS + tokens + EOS`；相邻文档被串成 token stream，再用一个 token 重叠切成固定长度 block：

```text
doc A: [BOS, A1, A2, EOS]
doc B: [BOS, B1, EOS]
stream: [BOS, A1, A2, EOS, BOS, B1, EOS, ...]
```

`input_ids` 与 `labels` 起初相同。模型内部用 `logits[:, :-1]` 对齐 `labels[:, 1:]`，padding label
在 collate 时写成 `-100` 并由 cross entropy 忽略。正式训练的完整 block 通常无需 padding；固定长度
validation reservoir 最后一批可能较小，但不会凭空添加监督 target。packing 允许跨文档预测 EOS 后的
BOS，这是当前明确选择，不是 document-isolated attention。

## Data and audit

`JsonlPretrainDataset` 是流式 `IterableDataset`，不会把语料整体加载进内存。train/validation 默认用
文本内容的 BLAKE2b hash 做稳定切分，同一文本及 exact duplicate 会落入同一侧。训练 block 经确定性
shuffle buffer 打乱。validation 在训练开始时完整扫描相应 stream，通过固定 seed 的 reservoir sampling
选出一组 block，之后每次评估复用同一组，避免只验证文件开头。

正式训练前建议生成完整数据报告：

```bash
uv run miniscale audit-pretrain-data \
  --data data/raw/minimind/pretrain/pretrain_t2t_mini.jsonl \
  --tokenizer data/tokenizer/minimind \
  --sequence-length 768 \
  --output artifacts/data-audit.json
```

报告会记录输入与 tokenizer SHA-256、JSON 错误、空文本、字符/token 数、stable split、packing 利用率和
exact duplicate 数。它只做审计，不会静默修改语料。当前仍未实现 near-duplicate 去重、质量分类、
来源 mixture 或 benchmark contamination 检测；这些字段会明确报告为 `false`。

## Default production recipe

`steps` 必须显式提供。其余默认值以 `PretrainOptions` 为唯一来源：

| 项目 | 默认值 |
| --- | ---: |
| decoder layers | 20 |
| micro batch | 1 sequence |
| gradient accumulation | 16 micro-batches |
| sequence length | 768 tokens |
| target tokens/update（world size 1） | 12,272 |
| AdamW peak LR / minimum LR | `3e-4` / `3e-5` |
| warmup | 200 optimizer steps |
| betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| weight decay / grad clip | `0.1` / `1.0` |
| validation | every 200 steps, up to 20 batches |
| checkpoint | every 500 steps, keep latest 3 |

总训练量由用户提供的 optimizer steps 决定，不由 dataset epoch 决定。理论 input tokens 是
`steps × batch_size × accumulation × sequence_length`；真正的 next-token targets 使用
`sequence_length - 1`，并同时记录为 `target_tokens_seen`。`pretrain_run.json` 保存解析后的 recipe、
planned tokens 和 tokens/parameter，避免只看到 micro batch 就误判训练规模。

新模型可用 `--num-hidden-layers` 调整 decoder 深度；默认 20 层约 63.6M 参数。层数会进入模型配置与
严格 resume identity，因此恢复时必须传入与原 run 相同的值，已有 checkpoint 不能跨层数加载。

## Optimizer, initialization, precision

AdamW 使用两组显式参数：Linear matrix weights 使用配置的 weight decay；RMSNorm、一维参数和 tied
embedding/lm head 使用零 weight decay。所有普通 Linear/Embedding 权重以 `N(0, 0.02)` 初始化，attention
output 与 MLP down residual projections 使用 `0.02 / sqrt(2 × num_layers)`，减少深度方向的残差方差累积。

默认 `--precision fp32`。支持 BF16 的 CUDA 设备可显式传 `--precision bf16`；训练、validation 和固定
generation probes 都使用 autocast，模型参数和 AdamW state 仍保持 FP32。BF16 不使用 GradScaler。
CPU 或不支持 BF16 的 CUDA 会在训练开始前报错，不会悄悄降级。当前未实现 FP16。

## Schedule, validation, logging, resume

LR 按 optimizer step 更新：200 step 线性 warmup，随后 cosine decay，在最后一次 planned update 使用
minimum LR。每个 accumulation micro-batch 的 loss 除以 accumulation steps；一次 update 后才 clip、
`optimizer.step()` 和 `scheduler.step()`。日志中的 `train_loss` 是该 optimizer update 内所有 target token
的加权平均，不是最后一个 micro-batch，也不是滑动平均。

本地 JSONL metrics 与可选 W&B 记录 loss、validation loss、perplexity、LR、pre-clip grad norm、是否
发生 clipping、input/target tokens、update time、tokens/s、samples/s 和 CUDA peak allocated memory。
目前不记录 MFU。

完整 checkpoint 保存模型、AdamW、scheduler、step、token/data 位置计数、best validation loss，以及
Python/NumPy/PyTorch/CUDA RNG。resume 会先校验数据、tokenizer、模型和轨迹相关参数的内容身份，然后
重放并跳过已消费的 micro-batches，最后恢复 checkpoint RNG。详细格式及旧 checkpoint 的一次性迁移见
[`checkpointing.md`](checkpointing.md)。

## Scaling boundary

当前正式路径是单进程、单 GPU/CPU 教学实现。尚未实现 DDP/FSDP、gradient checkpointing、
`torch.compile`、显式 FlashAttention backend 选择、dataloader state snapshot 或 MFU。扩展到多 GPU 时，
必须重新定义 global batch、data sharding、resume identity 和每 rank RNG，不能只包一层 DDP。
