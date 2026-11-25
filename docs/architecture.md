# 项目结构

MiniScale 在仓库根目录按模型、数据、训练阶段和辅助命令组织。阶段入口组织数据、训练和验证，共用模块处理参数、续训和产物。GRPO 与 Agent RL 共用 policy 训练循环，各自负责采样和评分。

| 位置 | 职责 |
| --- | --- |
| `model/model_miniscale.py` | 模型配置、decoder 和原生生成 |
| `model/tokenizer.py` | Hugging Face、SentencePiece、Byte tokenizer 及聊天模板 |
| `dataset/lm_dataset.py`、`dataset/sft_dataset.py`、`dataset/preference_dataset.py` | 流式预训练、assistant turn 索引和偏好配对 |
| `trainer/train_*.py` | 五个正式训练阶段与 SentencePiece 训练命令 |
| `trainer/trainer_utils.py`、`trainer/cli.py` | 设备、精度、seed、优化器、LR 和共用 CLI 参数 |
| `trainer/lm_utils.py`、`trainer/run_utils.py` | 语言模型数据准备和训练进度，共用指标与产物写入 |
| `trainer/checkpoint.py`、`trainer/tracking.py` | checkpoint 保存和恢复、W&B 及待传队列 |
| `trainer/rollout_engine.py`、`trainer/policy_training.py`、`trainer/rl_utils.py` | GRPO 和 Agent 的采样及评测，共用训练循环与 policy 更新 |
| `trainer/agent_env.py`、`trainer/rewards.py`、`trainer/eval_utils.py` | 计算器环境、数学奖励与生成质量评测 |
| `scripts/` | 审计、SFT 数据准备、tokenizer 检查、离线评测、环境检查和 CPU smoke |
| `eval_llm.py` | 调用 `scripts.inference` 的根目录推理命令 |

从仓库根运行 `uv run python trainer/train_pretrain.py --help` 或
`uv run python -m trainer.train_pretrain --help`。wheel 安装 `model`、`dataset`、
`trainer` 和 `scripts`，可在仓库外运行相同的模块入口，推理使用
`python -m scripts.inference`。默认 tokenizer、评估套件和质量策略随包分发，训练数据由用户提供。

## 依赖与实现原则

- 依赖方向为 `scripts → trainer → dataset → model`，允许跨层导入。Agent 的任务类型和计算器
  schema 筛选在 `dataset/rl_dataset.py`，工具执行和观测在 `trainer/agent_env.py`。
- 同一阶段的训练与验证共用本阶段的目标函数。可复用的运行机制、采样、生成评分和独立的 I/O
  在同级模块中维护。
- 包结构迁移后，checkpoint 读取层仍能识别原配置类。数据切分、checkpoint 格式或参数语义若有改变，
  应增加相应的兼容性测试，并在需要时提升格式版本。

`tests/` 按入口契约、模型、数据、训练和运行时分组，涵盖 mask、固定切分、五阶段权重交接和精确恢复。
