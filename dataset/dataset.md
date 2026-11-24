# 训练数据

训练命令从当前工作目录下的 `dataset/` 读取默认 JSONL。每行写一个 UTF-8 JSON 对象。
已有语料可直接用 `--data` 指定，例如 `--data data/raw/minimind/pretrain/pretrain_t2t_mini.jsonl`。
阶段支持独立验证文件时，可用 `--validation-data` 指定相同格式的 JSONL。

| 阶段 | 默认文件 | 每行的字段 |
| --- | --- | --- |
| Pretrain | `pretrain_t2t_mini.jsonl` | `text` 为非空文档文本，训练时流式拼接为 token block |
| SFT | `sft_t2t_mini.jsonl` | `conversations` 为 `role` / `content` 消息列表，支持 `reasoning_content` 和工具消息 |
| DPO | `dpo.jsonl` | `chosen`、`rejected` 为消息列表，共享相同的前缀，最后一条是各自的 assistant 回复 |
| GRPO | `agent_rl_math.jsonl` | `conversations` 提供最后一条 user prompt，`gt` 提供可验证答案列表 |
| Agent | `agent_rl_math.jsonl` | 使用相同的 `conversations` / `gt` 格式，可附带 `tools` 和 system 消息。工具定义按计算器能力筛选 |

SFT 根据 assistant turn 构建监督样本。派生文件可用 `sft_selection.target_positions`
指定训练用的 assistant 消息，位置从 0 开始。GRPO 与 Agent 对 prompt 进行稳定的 hash 切分；
Agent 工具观测进入上下文，训练 mask 仅覆盖模型动作。

用 `uv run python scripts/audit_data.py <pretrain|sft|dpo|grpo|agent> --help` 查看各阶段审计参数。
`scripts/prepare_data.py` 可生成清理后的 SFT JSONL，随仓库提供的质量策略在
`policies/sft_quality_v1.json`，SFT 生成评估套件在 `eval/sft_generation_v1.jsonl`。
本地训练 JSONL 和 `raw/` 通过 Git 和包构建规则排除。
