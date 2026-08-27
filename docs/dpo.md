# Direct preference optimization

MiniScale 的正式 DPO 路径直接承接完整 SFT checkpoint。每个偏好对必须包含共享 prompt，以及一个
chosen 和 rejected assistant 回复。两侧使用相同的近期 prompt context，只对目标回复计算 log-prob；
冻结 reference 始终是 DPO 启动时的 SFT policy。

## 训练前审计

审计只读取 MiniMind 数据并生成报告，不修改原始 JSONL：

```bash
uv run miniscale audit-dpo-data \
  --data data/raw/minimind/preference/dpo.jsonl \
  --tokenizer data/tokenizer/minimind \
  --max-length 512 \
  --sample-size 2000 \
  --output artifacts/dpo-data-audit.json
```

当前仓库数据的审计结果是 17,166 行、17,145 个有效 pair 和 21 个 chosen/rejected 完全相同的无效
pair。默认 prompt-hash 切分产生 16,277 个训练 pair 和 868 个验证 pair；无效 pair 会被计数并排除。
在 512 context 的 2,000 个全局随机样本中，53.9% 至少截断了一侧。该比例来自原始回答较长以及 SFT
模型 context 固定为 512；不能在 DPO 阶段把 `--max-length` 提高到 768。

## 数据与目标语义

训练启动时扫描 JSONL 并建立 byte-offset 索引，不把约 51 MiB 的文件整体载入内存。DataLoader 对完整
训练索引执行固定 seed 的全局 permutation。相同 prompt 的所有 pair 根据 canonical prompt hash 进入
同一 split，避免 train/validation 泄漏；也可通过 `--validation-data` 指定独立验证文件。

chosen/rejected 必须具有完全相同的 prompt。截断时两侧保留完全相同的近期 context，并分别保留回答
开头。DPO 使用标准的 completion log-prob 求和与 sigmoid objective；不按回答长度归一化。日志中的
reward 是 policy 相对冻结 SFT reference 的隐式 reward，因此应主要观察 reward accuracy、reward
margin 和 validation loss，而不是只比较 chosen/rejected 的原始 log-prob。

`--target-mode` 默认从 SFT checkpoint 继承。显式传入不同值会拒绝启动。当前
`artifacts/sft-v2-3050ti/sft.pt` 记录的是 `reasoning_and_response`。

## 20-step 冒烟测试

先验证数据、BF16、reference、validation 和 checkpoint：

```bash
uv run miniscale dpo \
  --data data/raw/minimind/preference/dpo.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/sft-v2-3050ti/sft.pt \
  --output artifacts/dpo-smoke-3050ti \
  --steps 20 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 5e-6 \
  --min-learning-rate 5e-7 \
  --beta 0.1 \
  --warmup-steps 2 \
  --validation-every 10 \
  --validation-batches 10 \
  --save-every 10 \
  --keep-last 2 \
  --generation-every 0 \
  --num-workers 0 \
  --device cuda
```

首步 policy 与 reference 相同，所以 DPO loss 应接近 `log(2) = 0.6931`，reward margin 为 0。后续 loss
应保持有限，reward margin 通常逐渐转正。确认 `reference.pt`、`best.pt`、周期 checkpoint 和最终
`dpo.pt` 均能生成后，再开始正式 run。新训练拒绝覆盖已有输出目录。

## 3050 Ti 正式训练

16,277 个训练 pair 配合有效 batch 16，1,000 optimizer steps 约等于遍历一轮训练数据：

```bash
uv run miniscale dpo \
  --data data/raw/minimind/preference/dpo.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/sft-v2-3050ti/sft.pt \
  --output artifacts/dpo-v2-3050ti \
  --steps 1000 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 5e-6 \
  --min-learning-rate 5e-7 \
  --beta 0.1 \
  --warmup-steps 50 \
  --validation-every 100 \
  --validation-batches 100 \
  --save-every 200 \
  --keep-last 3 \
  --generation-every 500 \
  --generation-max-new-tokens 96 \
  --num-workers 0 \
  --device cuda \
  --wandb \
  --wandb-project MiniScale
```

如果未安装或未登录 W&B，删除 `--wandb` 和 `--wandb-project`。DPO 将 policy 和 reference 参数保留
为 FP32，policy/reference forward 使用 BF16 autocast，log-softmax 和 log-prob reduction 使用 FP32。

## 5070 12GB

先使用 `--batch-size 2 --gradient-accumulation 8`。显存稳定后可以尝试 `4 × 4`，两者都保持每个
optimizer update 为 16 个 preference pair，因此不需要修改 LR、beta、steps 或 seed。验证批次数是
batch 数：`batch-size 4 --validation-batches 100` 会验证最多 400 个固定 pair。

## Checkpoint 与恢复

输出目录包含：

```text
artifacts/dpo-v2-3050ti/
├── checkpoints/step_XXXXXXXX.pt
├── generations/step_XXXXXXXX.json
├── dpo_metrics.jsonl
├── dpo_run.json
├── reference.pt
├── best.pt
└── dpo.pt
```

`reference.pt` 是只保存一次的不可变冻结 reference，避免在每个周期 checkpoint 中重复一份模型权重。
精确 resume 需要保留整个输出目录。周期、best 和 final checkpoint 保存 policy、AdamW、scheduler、
RNG、数据进度、SFT 父权重身份和 reference 内容身份。

恢复时保留首次正式命令中的所有轨迹参数，只把 `--checkpoint` 替换为 `--resume`，输出目录不变：

```bash
uv run miniscale dpo \
  --data data/raw/minimind/preference/dpo.jsonl \
  --tokenizer data/tokenizer/minimind \
  --resume artifacts/dpo-v2-3050ti/checkpoints/step_00000200.pt \
  --output artifacts/dpo-v2-3050ti \
  --steps 1000 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 512 \
  --precision bf16 \
  --learning-rate 5e-6 \
  --min-learning-rate 5e-7 \
  --beta 0.1 \
  --warmup-steps 50 \
  --validation-every 100 \
  --validation-batches 100 \
  --save-every 200 \
  --keep-last 3 \
  --generation-every 500 \
  --generation-max-new-tokens 96 \
  --num-workers 0 \
  --device cuda \
  --wandb \
  --wandb-project MiniScale
```

`--steps 1000` 是原计划总步数，不是剩余步数。若要改变总 steps、LR、beta 或数据策略，应从原 SFT
checkpoint 使用新输出目录重新开始 DPO 实验，不应把轨迹变化伪装成精确 resume。

最终 `dpo.pt` 保留顶层 `config` 与 `model`，可以直接用于 `generate` 和下一阶段 GRPO。
