# MiniScale

MiniScale 是一个从零实现、可测试的“小模型到 Agent RL”学习项目。它借鉴了
[MiniMind](https://github.com/jingyaogong/minimind) 的完整训练路线和
[MokioMind](https://github.com/Wood-Q/MokioMind) 的手写学习方式，但模型、数据管线、
GRPO 目标和工具环境均在本仓库独立实现，不是对上游源码的改名复制。

当前目标是 **MVP 级流水线正确性**：在一台笔记本上用两个样本刻意过拟合，验证每个阶段的
数据流、损失、checkpoint 和阶段衔接。它不代表模型具备泛化能力。

## 已打通的流水线

```text
UTF-8 文本 -> Pretrain (next-token loss)
            -> SFT (仅 assistant token loss)
            -> RLVR/GRPO (规则奖励 + group advantage + clip + reference KL)
            -> Agent GRPO (多轮 tool call / observation / action-only loss)
            -> pipeline_manifest.json + 各阶段 checkpoint
```

模型部分原生 PyTorch 实现了 RMSNorm、RoPE、GQA、SDPA causal attention、SwiGLU、
tied embeddings；Tokenizer 暂用确定性的 byte tokenizer，从而不下载外部模型就能复现实验。

## 立即运行

项目固定 Python `>=3.11,<3.13`，`.python-version` 为 3.12。请始终通过 `uv run` 或
激活 `.venv` 运行；终端裸执行 `python` 若仍显示 3.14，只表示 base 环境在 PATH 前面，
不代表本项目虚拟环境异常。

```bash
uv sync
uv run miniscale doctor
uv run python -m unittest discover -s tests -v
uv run miniscale pipeline --device cpu --output artifacts/mvp
```

有可用 CUDA 时可将 `cpu` 改为 `cuda`。RTX 3050 Ti 4GB 适合本仓库 smoke 配置，
不适合直接训练默认的较大配置或长序列。

也可以分阶段阅读和运行：

```bash
uv run python trainer/train_pretrain.py
uv run python trainer/train_sft.py
uv run python trainer/train_grpo.py
uv run python trainer/train_agent.py
```

输出在 `artifacts/`，该目录已被 Git 忽略。流水线清单会记录各阶段 loss、reward、耗时和
checkpoint 路径。

## 建议学习顺序

1. `config.py`、`tokenizer.py`、`model.py`：亲手推导张量 shape，确认 causal test 为什么成立。
2. `data.py`、`training/pretrain.py`：理解 shift label、padding 和 next-token objective。
3. `training/sft.py`：检查 `-100` mask，确保 user/system/tool 文本不成为监督目标。
4. `training/grpo.py`：先看 reward group normalization，再看 ratio clipping 和 reference KL。
5. `agent_env.py`、`training/agent_rl.py`：跟踪一次两轮 trajectory，确认 observation 进入上下文但
   action mask 为 0。
6. 修改一个组件并补测试，例如新增 `search` mock 工具、格式奖励或离线 eval；这才会逐渐变成
   你的工程，而不是 MiniMind 的复刻。

## 与工业训练栈的边界

这个 MVP 刻意保留了工业界最重要的语义：阶段化 checkpoint、assistant/action masking、
old/reference policy、可验证 reward、受限工具执行和端到端测试。但真正扩大训练规模前还需要：

- 用 SentencePiece/BPE 和正式数据集替换 byte tokenizer 与内置样本，并做去重、质量过滤、污染检测；
- 加入 BF16、gradient accumulation/checkpointing、FlashAttention、sequence packing、FSDP/DeepSpeed；
- 将 rollout 与 learner 解耦，用 vLLM/SGLang 一类推理服务异步采样，并处理 policy weight 同步；
- 对工具执行使用进程/容器级 sandbox、超时、资源限额和审计，而不仅是本 MVP 的 AST 白名单；
- 建立固定 held-out eval、pass@k、tool-call success、KL/entropy、吞吐和显存监控；
- 对 reward hacking、长度偏置、全组同分和训练/推理 chat template 不一致做专项回归。

求职展示时，应把 `pipeline_manifest.json`、测试、消融实验和失败案例作为证据；“能运行”只是
第一层，能解释 reward、mask、KL、采样吞吐和评测可信度才是 Agentic RL 岗位更看重的部分。

## 目录

```text
src/miniscale/          核心模型、数据、环境与训练算法
trainer/                各阶段可直接运行的薄入口
tests/                  算法语义与端到端回归测试
artifacts/              本地 checkpoint/manifest（不提交）
```

## 阶段提交

本仓库按可验证里程碑提交：预训练、SFT、GRPO、Agent RL、最终 CLI/文档。可用
`git log --oneline` 审核历史，用 `git show <commit>` 查看某阶段完整改动。
