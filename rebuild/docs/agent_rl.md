# Calculator Agent-RL

Agent-RL 是可选阶段：只有最终产品需要工具调用时才应在 GRPO 后运行。当前环境只注册受限计算器；
天气、汇率、翻译等工具没有真实 sandbox 和 verifier，不会被模型看到，也不能使用混合
`agent_rl.jsonl` 直接训练。

## 数据与工具能力审计

```bash
uv run miniscale audit-agent-data \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --validation-fraction 0.05 \
  --output artifacts/agent-data-audit.json
```

当前数据的 19,430 个去重任务被切分为 18,484 train / 946 validation。原始行会同时声明一个计算器和
一个随机的非计算器工具；审计报告会按名称统计这些 schema，训练加载器只保留可执行的计算器 schema。
若某行只声明不支持的工具，它会被明确计入无效数据，而不是静默进入训练。

工具协议要求恰好一个完整的 `<tool_call>{...}</tool_call>` JSON block。环境区分 malformed call、
unsupported tool、invalid arguments 和 execution error。计算仅允许 AST 白名单中的数值四则、取模、
整除和受限幂运算；任意 Python 调用、过长表达式和过大结果都会失败。

工具 observation 会进入下一轮上下文，但对应 action mask 为 0，不参与 policy loss。奖励由最终答案
正确性、合法工具调用 bonus、额外答案惩罚和非法调用惩罚组成；工具参数里的数字不能冒充最终答案。

## 冒烟测试

```bash
uv run miniscale agent-rl \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/grpo-v2-3050ti/best.pt \
  --output artifacts/agent-rl-smoke \
  --steps 10 --batch-size 1 --group-size 2 \
  --policy-epochs 2 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --reference-device cpu \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --warmup-steps 2 \
  --validation-every 5 --validation-prompts 10 \
  --save-every 5 --keep-last 2 --device cuda
```

重点观察 success rate、tool-call rate、invalid-call rate、mean turns、reward、KL 和
zero-advantage-group rate。工具调用率上升但 success 不升，通常代表模型只学会了格式，没有学会利用
observation 得到最终答案。

## 正式训练起点

3050 Ti 建议从 `batch-size 1 × group-size 2`、`max-turns 3`、每轮 64 token、CPU reference 开始；
5070 12GB 可从 `batch-size 1 × group-size 4`、`max-turns 4`、每轮 96 token、GPU reference 开始。
Agent trajectory 比单轮 GRPO 长得多，应先增加 group size，再尝试增加 batch size。

一个保守的 3050 Ti 配置：

```bash
uv run miniscale agent-rl \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/grpo-v2-3050ti/best.pt \
  --output artifacts/agent-rl-v2-3050ti \
  --steps 500 --batch-size 1 --group-size 2 \
  --policy-epochs 2 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --reference-device cpu \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --warmup-steps 20 \
  --temperature 1.0 --top-k 50 --beta 0.01 --clip-epsilon 0.2 \
  --validation-every 50 --validation-prompts 50 \
  --save-every 50 --keep-last 3 --device cuda \
  --wandb --wandb-project MiniScale
```

恢复方式与 GRPO 相同：保持所有轨迹参数不变，将 `--checkpoint` 替换为 `--resume`，并继续使用原输出
目录。最终优先比较 `best.pt`，不要默认使用最后一步。

## 真实工具推理

`generate --calculator` 会执行完整的模型→工具→observation→模型循环，不再只是注入工具说明：

```bash
uv run miniscale generate \
  --checkpoint artifacts/agent-rl-v2-3050ti/best.pt \
  --tokenizer data/tokenizer/minimind \
  --prompt "请计算 7109*2920，只给最终结果。" \
  --calculator --max-turns 3 --temperature 0 --max-new-tokens 64
```

JSON 输出包含最终 response、完整 transcript、合法/非法工具调用数和轮数，便于验收状态机行为。
