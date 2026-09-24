# Cross-stage RL evaluation

`miniscale evaluate` lets the same immutable validation prompts compare checkpoints from different stages. It fully
scans and fingerprints the source data, uses the same prompt-hash split as training, takes a fixed global sample and
runs greedy generation. The JSON report records checkpoint, tokenizer and data identities so results remain auditable.

Compare DPO with GRPO on direct mathematical answers:

```bash
uv run miniscale evaluate \
  --kind grpo \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/dpo-v2-3050ti/dpo.pt \
  --checkpoint artifacts/grpo-v2-3050ti/best.pt \
  --prompts 100 --max-new-tokens 96 \
  --precision bf16 --device cuda \
  --output artifacts/eval-dpo-vs-grpo.json
```

Compare the selected GRPO model with Agent-RL using the real calculator loop:

```bash
uv run miniscale evaluate \
  --kind agent \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/grpo-v2-3050ti/best.pt \
  --checkpoint artifacts/agent-rl-v2-3050ti/best.pt \
  --prompts 50 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --device cuda \
  --output artifacts/eval-grpo-vs-agent.json
```

GRPO 主要比较 `validation_exact_match` 和 `validation_reward`；Agent 主要比较
`validation_success_rate`、合法/非法工具调用率和平均轮数。最终发布应选择验证表现最佳且通用聊天抽查
没有明显退化的 checkpoint，而不是机械选择训练最后一步。当前命令覆盖可验证数学和计算器行为；开放
式聊天质量仍需单独准备人工或经过校准的 benchmark，不能用规则数字奖励替代。
