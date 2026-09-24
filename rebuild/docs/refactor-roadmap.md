# MiniScale 渐进式重构路线

这份路线用于当前学习分支；具体重构位于仓库根目录的 `rebuild/`。目标是让训练改动有可复现配方、固定评估和明确的效率记录，同时保留单卡项目容易阅读的特点。

## 参照项目

- [nanochat](https://github.com/karpathy/nanochat)：核心模型、数据、推理与 checkpoint 分开；阶段脚本和固定运行配方放在外层，并把质量与训练耗时一起比较。借鉴运行入口与基准，不照搬多 GPU 假设。
- [LitGPT](https://github.com/Lightning-AI/litgpt)：可复用训练 recipe 和统一 CLI。借鉴配方文件及参数记录，保留本仓库可读的阶段训练循环。
- [TorchTitan](https://github.com/pytorch/torchtitan)：实验先放在独立区域，经过验证后并入核心。借鉴实验隔离方式；本项目暂不引入分布式框架结构。
- [nanoGPT](https://github.com/karpathy/nanoGPT)：模型和训练循环简洁，但上游 README 已将其标为旧项目，并推荐 nanochat。

## 当前边界

- 正式入口是 `miniscale` CLI；`trainer/`、`model/`、`dataset/` 是 smoke 或兼容入口。
- 各阶段训练实现已迁到 `rebuild/src/miniscale/training/{pretrain,sft,dpo,grpo,agent_rl}/`；旧的 `stages/`、`configs/`、`evaluators/` 和 `objectives/` 仅做兼容转发。训练循环仍较长，后续按语义逐项缩小。
- 预训练已有 token 吞吐、显存峰值和验证 loss；SFT 与 RL 已有质量指标。缺少统一的实验对照记录、独立的 4090 基准及固定的数据质量门槛。
- 正式 CLI 目前只能直接调整层数。宽度、FFN 维度和 head 数仍固定在 `small_64m()` 中。
- 原始训练 JSONL 不在仓库中，因此无法在本机判断数据上限或正式训练时长。

## 目标结构

目录按职责组织，阶段内部保留完整、容易跟读的训练流程。下列阶段目录已在 `rebuild/` 中建立。

```text
src/miniscale/
├── model.py                  # 模型前向；需要时再拆
├── tokenizer.py              # token 与 chat 格式；需要时再拆
├── data/                     # 数据格式、切分、packing、审计
├── training/
│   ├── core/                 # 设备、优化器、checkpoint、指标等共享机制
│   ├── pretrain/             # config、runner、eval、resume
│   ├── sft/                  # config、runner、eval
│   ├── dpo/                  # config、objective、runner、eval
│   ├── grpo/                 # config、objective、rollout、runner、eval
│   └── agent_rl/             # 环境、rollout、runner、eval
├── inference.py
├── evaluation.py             # 跨阶段固定套件与对照报告
└── cli/                      # 薄解析与分发，不包含训练算法
recipes/                      # 可提交的模型与训练配方
experiments/                  # 实验说明；大产物仍在 artifacts/
tests/                        # 语义、恢复、CLI 合约和集成测试
docs/
```

`training/core/` 只放确实被多个阶段共享的机制。不建立隐藏 DPO、GRPO 或 Agent RL 算法差异的万能 Trainer。旧 Python 导入路径与 CLI 在迁移期间用薄转发保留；改变 checkpoint schema 或训练语义时显式更新版本与测试。

## 每次改动的实验记录

先写清一个假设，并记录：基线 Git commit、候选 commit、模型配置、tokenizer 与数据指纹、seed、实际训练 token/步数、精度、显卡、峰值显存、训练 tok/s、完整耗时、固定验证质量以及 checkpoint。比较质量时保持训练 token 与评估集一致；比较效率时保持模型、数据、batch、上下文与精度一致。无显著改善的候选也记录下来。

| 阶段 | 主要质量指标 | 主要效率指标 | 语义门槛 |
| --- | --- | --- | --- |
| 预训练 | 固定验证 loss、perplexity、分领域 loss | 训练 tok/s、峰值显存、总耗时 | next-token shift、packing、resume 一致 |
| SFT | 固定生成套件、人工抽查、格式遵循 | 监督 token/s、峰值显存 | assistant mask、截断正确 |
| DPO | held-out preference accuracy、reward margin | pair/s、峰值显存 | 共享 prompt、冻结 reference |
| GRPO/Agent RL | held-out exact match、成功率、无效工具率 | rollout/s、update 时间、显存 | action mask、old/reference policy、奖励正确 |

训练 loss 不能单独代表成功率；单步 tok/s 也不能代表端到端效率。先固定评估集与计时边界，再做性能优化。

## 迁移顺序与验收

0. **锁定基线**：运行现有测试，记录 `doctor`、smoke pipeline、正式 CLI 参数和 checkpoint 格式。准备固定的 4090 短跑配方。这一步不修改训练数学。
1. **可比较的 recipe 与报告**：给模型结构和各阶段训练建立可提交配方，CLI 解析并把最终配置写入现有 run manifest。先支持预训练，再扩展其他阶段。增加从多个 run manifest/metrics 生成对照表的轻量工具。验收：同一配方重复运行时身份明确，改变一个参数能在报告中看出。
2. **拆分预训练阶段（已完成结构迁移）**：从 `training/stages/pretrain.py` 逐项搬出评估、resume 身份和训练循环，形成 `training/pretrain/`。每次只迁移一个职责，保留旧导入转发。验收：公开 API、CLI、完整 checkpoint 与中断恢复测试均通过；固定 seed 的短跑 loss 和权重一致。
3. **单卡性能实验**：在 4090 上先测 baseline，再分别实验原生 GQA/SDPA、数据预处理或缓存、推理 KV cache。仅在显存限制实际出现时测试 activation checkpointing。每次单独改一个因素，报告 tok/s、显存、固定质量与完整运行时间。优化失败则保留实验记录，不把复杂度并入主线。
4. **提高数据与质量上限**：在已有 exact duplicate 审计上增加近重复、来源配比和固定分领域验证；生成套件覆盖中文、英文、代码、数学及格式遵循。根据数据审计结果确定模型规模和训练 token 预算。
5. **迁移后续阶段（已完成结构迁移）**：SFT、DPO、GRPO、Agent RL 的实现已迁入各自目录。继续缩小较长的 runner，优先处理生成与 rollout 的推理成本；性能或质量优化仍需独立实验验证。

## 第一轮操作

1. 阅读本文件和 `docs/architecture.md`，确认各目录职责。
2. 运行 `uv run python -m unittest discover -s tests -q`，保存测试基线。
3. 有原始数据后运行 `miniscale audit-pretrain-data`。4090 到位后用独立短测目录记录吞吐与显存。
4. 下一步实现预训练 recipe 和对照报告；训练实现当前已迁入阶段目录。

## 当前进度

- 重构项目位于 `rebuild/`；外层项目作为原始参照。
- 数据层已拆出 JSONL 读取、batch/验证抽样、流式预训练和 smoke SFT 数据集。长文档 packing 改为按偏移读取，并加入跨 block/文档边界测试。
- 预训练恢复身份、run manifest 和固定生成评估已离开主训练文件；旧导入入口继续可用。
- 五个训练阶段已有独立目录；GRPO 与 Agent RL 的 rollout 和 eval 已从 runner 拆出，旧导入路径继续可用。
- 下一步是预训练 recipe 与实验对照报告，然后在 4090 上建立真实速度基线。
