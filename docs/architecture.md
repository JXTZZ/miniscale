# 项目结构

MiniScale 在仓库根目录按模型、数据、训练阶段和辅助命令组织。阶段入口组织数据、训练和验证，共用模块处理参数、续训和产物。GRPO 与 Agent RL 共用 policy 训练循环，各自负责采样和评分。

| 位置 | 职责 |
| --- | --- |
| `model/model_miniscale.py`、`model/lora.py` | 模型配置、decoder、可选 KV cache 和低秩适配器 |
| `model/tokenizer.py` | Hugging Face、SentencePiece、Byte tokenizer 及聊天模板 |
| `dataset/lm_dataset.py`、`dataset/sft_dataset.py`、`dataset/preference_dataset.py` | 流式预训练、assistant turn 索引和偏好配对 |
| `trainer/train_*.py` | 五个正式训练阶段与 SentencePiece 训练命令 |
| `trainer/cli.py` | 共用参数、checkpoint 来源和命令适配 |
| `trainer/checkpoint.py`、`trainer/runtime/` | 状态序列化、安全更新边界、设备与优化器、指标和追踪 |
| `trainer/lm_utils.py`、`trainer/sft_eval.py`、`trainer/preference_objective.py` | 语言模型数据准备和训练进度、SFT 选模、DPO 目标 |
| `trainer/rl/` | 策略配置和训练、GRPO/CISPO 目标、采样、计算器环境与评测 |
| `scripts/` | 审计、SFT 数据准备、tokenizer 检查、离线评测、环境检查和 CPU smoke |
| `eval_llm.py` | 调用 `scripts.inference` 的根目录推理命令 |

从仓库根运行 `uv run python trainer/train_pretrain.py --help` 或
`uv run python -m trainer.train_pretrain --help`。wheel 安装 `model`、`dataset`、
`trainer` 和 `scripts`，可在仓库外运行相同的模块入口，推理使用
`python -m scripts.inference`。默认 tokenizer、评估套件和质量策略随包分发，训练数据由用户提供。

## 依赖与实现原则

- 依赖方向为 `scripts → trainer → dataset → model`，允许跨层导入。Agent 的任务类型和计算器
  schema 筛选在 `dataset/rl_dataset.py`，工具执行和观测在 `trainer/rl/environment.py`。
- CLI 把命令参数交给阶段入口。运行时处理保存和记录，RL 采样只读取生成选项，供训练、推理和评测共用。
- 包结构迁移后，checkpoint 读取层仍能识别原配置类。数据切分、checkpoint 格式或参数语义若有改变，
  应增加相应的兼容性测试，并在需要时提升格式版本。

`tests/` 按入口契约、模型、数据、训练和运行时分组，涵盖 mask、固定切分、五阶段权重交接和精确恢复。
