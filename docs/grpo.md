# Verifiable GRPO

GRPO 承接 DPO checkpoint，默认只训练 `agent_rl_math.jsonl` 中具有确定 `gt` 的数学任务。开放式
`rlaif.jsonl` 没有可信 verifier，不能直接传给本训练器。

## 数据审计

```bash
uv run miniscale audit-grpo-data \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --validation-fraction 0.05 \
  --sample-size 2000 \
  --output artifacts/grpo-data-audit.json
```

当前 20,000 行数据包含 19,430 个去重后的有效任务和 570 个重复任务；稳定 prompt-hash 切分得到
18,484 个训练任务和 946 个验证任务。`--data-limit` 在全量扫描后做固定 seed 的全局抽样，不再取文件
开头 N 行。原始文件不会被修改。

奖励只解析模型可见的最终回答，`<tool_call>` 参数不计入答案。正确答案全集必须与预测数字全集一致
才能得到 exact/format bonus；额外猜测数字会被惩罚，避免通过罗列大量候选答案投机。

## 训练语义

一个 `step` 表示采集 `batch_size × group_size` 条 rollout，然后执行 `policy_epochs` 次 optimizer
update。旧策略 log-prob 在采样后、任何更新前缓存；同一 rollout 的后续 policy epoch 才会产生真实
ratio 偏移和 clipping。损失先在每条序列的 action token 内平均，再在序列间平均，避免长回答获得更高
权重。prompt 和 padding 不参与策略梯度。

policy 参数和 AdamW state 保持 FP32；`--precision bf16` 只对 CUDA forward 使用 autocast，
log-softmax、KL、reward 和 advantage 保持 FP32。`--reference-device cpu` 可减少显存，但 reference
scoring 会更慢。

## 冒烟测试

```bash
uv run miniscale grpo \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/dpo-v2-3050ti/dpo.pt \
  --output artifacts/grpo-smoke \
  --steps 10 --batch-size 1 --group-size 2 \
  --policy-epochs 2 --max-new-tokens 64 \
  --precision bf16 --reference-device cpu \
  --learning-rate 1e-5 --min-learning-rate 1e-6 --warmup-steps 2 \
  --validation-every 5 --validation-prompts 10 \
  --save-every 5 --keep-last 2 --device cuda
```

确认 loss/reward/KL 有限，并生成 `reference.pt`、`best.pt`、周期 checkpoint、`last.pt` 和 `rl.pt`
后再启动正式训练。`zero_advantage_group_rate` 很高表示同组采样几乎总是同分，此时应先检查模型输出、
temperature、group size 和奖励覆盖，不应靠提高学习率强行训练。

## 3050 Ti 建议

```bash
uv run miniscale grpo \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/dpo-v2-3050ti/dpo.pt \
  --output artifacts/grpo-v2-3050ti \
  --steps 1000 --batch-size 1 --group-size 4 \
  --policy-epochs 2 --max-new-tokens 96 \
  --precision bf16 --reference-device cpu \
  --learning-rate 1e-5 --min-learning-rate 1e-6 --warmup-steps 40 \
  --temperature 1.0 --top-k 50 --beta 0.01 --clip-epsilon 0.2 \
  --validation-every 100 --validation-prompts 100 \
  --save-every 100 --keep-last 3 --device cuda \
  --wandb --wandb-project MiniScale
```

若显存仍不足，先把 `--max-new-tokens` 降到 64，再把 group size 降到 2。不要先降低 beta 或关闭
reference KL。

## 5070 12GB 建议

先使用 `--batch-size 2 --group-size 4 --reference-device same --max-new-tokens 128`。稳定后可尝试
group size 6 或 8；每次只改变一个吞吐参数，并比较 validation reward、zero-advantage rate 和显存峰值。
增加 rollout 数不要求线性增加 learning rate。

## Checkpoint 与恢复

`reference.pt` 是启动 policy 的不可变快照。`best.pt`、`last.pt`、`rl.pt` 和周期 checkpoint 都包含
policy、AdamW、scheduler、RNG、任务游标、rollout/token 计数、数据指纹和 reference 指纹。精确恢复
必须保留整个输出目录。支持 hard link 的文件系统上 `last.pt` 是 `rl.pt` 的磁盘零拷贝别名：

```bash
uv run miniscale grpo \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --resume artifacts/grpo-v2-3050ti/checkpoints/step_00000200.pt \
  --output artifacts/grpo-v2-3050ti \
  --steps 1000 --batch-size 1 --group-size 4 \
  --policy-epochs 2 --max-new-tokens 96 \
  --precision bf16 --reference-device cpu \
  --learning-rate 1e-5 --min-learning-rate 1e-6 --warmup-steps 40 \
  --temperature 1.0 --top-k 50 --beta 0.01 --clip-epsilon 0.2 \
  --validation-every 100 --validation-prompts 100 \
  --save-every 100 --keep-last 3 --device cuda
```

恢复时 `--steps` 仍是原计划总步数。改变数据、seed、采样、目标、精度或优化器参数会被拒绝。
