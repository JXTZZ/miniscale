# Supervised fine-tuning

MiniScale 的正式 SFT 路径把一个 conversation 展开为多个 assistant-turn 样本。每个样本保留截至
当前 assistant 回复的对话上下文，但只监督当前回复；user、system、tool observation、历史 assistant
回复和 role header 都使用 `-100` mask。模型仍在内部完成 next-token shift。

## 训练前审计

先完整扫描结构、exact duplicate、稳定切分和身份字符串，再对全局随机样本计算 token 长度与截断：

```bash
uv run miniscale audit-sft-data \
  --data data/raw/minimind/sft/sft_t2t_mini.jsonl \
  --tokenizer data/tokenizer/minimind \
  --max-length 512 \
  --identity-pattern MiniMind \
  --identity-pattern jingyaogong \
  --output artifacts/sft-data-audit.json
```

审计只生成报告，不改写原始数据，也不会根据身份字符串自动过滤。正式训练默认按 canonical
conversation identity 跳过 exact duplicate，并把去重数量写入 `sft_run.json`。如果需要过滤或重写
身份回答，应使用独立数据准备命令生成新的 JSONL，而不是在训练循环里静默修改：

```bash
uv run miniscale prepare-sft-data \
  --data data/raw/minimind/sft/sft_t2t_mini.jsonl \
  --exclude-pattern MiniMind \
  --exclude-pattern jingyaogong \
  --output data/derived/sft/sft_v2.jsonl
```

该命令默认 exact dedup，拒绝覆盖源文件或已有输出，并同时写入
`sft_v2.jsonl.manifest.json`，记录源/输出 SHA-256、过滤规则和各类计数。是否过滤身份样本属于数据
策略；在审阅抽样内容前不要仅凭字符串计数直接执行。

## 数据顺序、切分和截断

启动时会扫描 JSONL，建立 assistant-turn 的 byte-offset 索引。DataLoader 对完整索引执行固定 seed
的全局 permutation，因此短训练不会只读取文件前部。相同 seed、数据和 worker 数得到相同顺序。

默认用 canonical conversation hash 将 0.5% conversation 放入 validation；exact duplicate 总是属于
同一个 split。也可以通过 `--validation-data` 使用独立 JSONL。程序固定随机抽取 validation 样本并在
每次验证时复用；`--validation-batches 100 --batch-size 1` 就是固定 100 个随机 held-out 样本。

样本超过 context 时，程序保留最近的 prompt context 和 assistant 回复开头，不再无条件保留最后
512 token。默认至少保留 32 个上下文 token；如果回复本身过长，会截断回复尾部。训练样本绝不会从
supervised token 开始。

## Reasoning 监督策略

- `--target-mode reasoning_and_response`：监督 `<think>`、reasoning、最终回复和 assistant 结束符；
- `--target-mode response_only`：目标 assistant 的 reasoning 会从输入中移除，只监督最终回复和结束符。

默认使用 `reasoning_and_response`，与 MiniMind 数据原有语义一致。该选择属于严格 resume identity，
不能在同一个 run 中途切换。后续 DPO 必须使用相同的 `--target-mode`；DPO 已复用同一个
assistant-turn mask 和安全截断实现。

## 从旧 10k 预训练权重开始

旧预训练 `best.pt` 可以作为 SFT 的初始化权重。它不是 SFT optimizer checkpoint，所以应使用
`--checkpoint`，不能使用 `--resume`。旧权重 context 为 512，SFT 不能把 `--max-length` 提高到 768。

先进行 20 step 验证：

```bash
uv run miniscale sft \
  --steps 20 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 2e-5 \
  --min-learning-rate 2e-6 \
  --warmup-steps 2 \
  --validation-every 10 \
  --validation-batches 10 \
  --save-every 10 \
  --keep-last 2 \
  --generation-every 0 \
  --num-workers 0 \
  --device cuda \
  --checkpoint artifacts/pretrain-shuffled-2/best.pt \
  --output artifacts/sft-smoke-3050ti
```

确认 loss 有限、`best.pt`/`sft.pt` 可加载、validation 可重复后，再启动笔记本 3050 Ti 正式实验：

```bash
uv run miniscale sft \
  --steps 3000 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 2e-5 \
  --min-learning-rate 2e-6 \
  --warmup-steps 100 \
  --validation-every 200 \
  --validation-batches 100 \
  --save-every 500 \
  --keep-last 3 \
  --generation-every 1000 \
  --num-workers 0 \
  --device cuda \
  --wandb \
  --wandb-project MiniScale \
  --checkpoint artifacts/pretrain-shuffled-2/best.pt \
  --output artifacts/sft-v2-3050ti
```

5070 12GB 可以先尝试 `--batch-size 4 --gradient-accumulation 4`，保持每个 optimizer update 仍为
16 个样本；若显存不足退到 `2 × 8`。不要因为更换显卡而修改 LR、数据 seed 或总 steps，除非明确
开始一个新实验。BF16 会在启动时检查硬件支持，模型参数与 AdamW state 保持 FP32。

## Checkpoint 与恢复

输出目录包含：

```text
artifacts/sft-v2-3050ti/
├── checkpoints/step_XXXXXXXX.pt
├── generations/step_XXXXXXXX.json
├── sft_metrics.jsonl
├── sft_run.json
├── best.pt
└── sft.pt
```

`best.pt`、周期 checkpoint 和最终 `sft.pt` 都包含模型、AdamW、scheduler、step、数据进度、
Python/NumPy/PyTorch/CUDA RNG、输入/监督 token 计数和严格 resume identity。`sft.pt` 仍保留顶层
`config` 与 `model`，所以 generate、DPO 和旧的 `load_checkpoint` 调用方式不变。

精确恢复时，`--steps` 表示原计划总步数，所有影响训练轨迹的参数必须和首次命令一致：

```bash
uv run miniscale sft \
  --steps 3000 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 2e-5 \
  --min-learning-rate 2e-6 \
  --warmup-steps 100 \
  --validation-every 200 \
  --validation-batches 100 \
  --save-every 500 \
  --keep-last 3 \
  --generation-every 1000 \
  --num-workers 0 \
  --device cuda \
  --wandb \
  --wandb-project MiniScale \
  --resume artifacts/sft-v2-3050ti/checkpoints/step_00000500.pt \
  --output artifacts/sft-v2-3050ti
```

如果想改变总 steps、LR 或数据策略，应把已有 SFT checkpoint 当作新实验的 `--checkpoint`；这会重置
optimizer/scheduler，并在新 manifest 中记录父权重内容指纹。新训练拒绝覆盖已有输出，精确 resume
会先校验模型、数据、tokenizer、mask、截断、采样、optimizer、schedule、batch、seed 和精度。

## 指标

训练日志和可选 W&B 会记录 train/validation loss、perplexity、validation token accuracy、LR、
gradient norm、input/supervised tokens、examples seen、吞吐和 CUDA peak memory。梯度累计按有效
supervised token 加权，所以长短回答不会因 micro-batch 边界获得不同权重。固定 greedy generation
用于观察中文、英文、代码和身份回答，但不参与 best checkpoint 选择。
