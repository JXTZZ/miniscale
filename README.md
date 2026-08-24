# MiniScale

MiniScale 是一个从零实现、可测试的“小模型到 Agent RL”学习项目。它借鉴了
[MiniMind](https://github.com/jingyaogong/minimind) 的完整训练路线和
[MokioMind](https://github.com/Wood-Q/MokioMind) 的手写学习方式，但模型、数据管线、
GRPO 目标和工具环境均在本仓库独立实现，不是对上游源码的改名复制。

仓库同时保留秒级 smoke pipeline，并提供约 64M 参数模型的真实 JSONL 训练入口。真实入口
直接读取本地数据，不会把十几 GB 数据一次性装入内存；训练步数必须显式指定，避免误启动
长任务。

## 已打通的流水线

```text
JSONL 文本 -> SentencePiece -> Pretrain (next-token loss)
            -> SFT (仅 assistant token loss)
            -> DPO (chosen/rejected + frozen reference)
            -> RLVR/GRPO (可验证数学奖励 + group advantage + clip + reference KL)
            -> Agent GRPO (多轮 tool call / observation / action-only loss)
            -> 各阶段 checkpoint + JSONL metrics
```

模型部分原生 PyTorch 实现了 RMSNorm、RoPE、GQA、SDPA causal attention、SwiGLU、
tied embeddings。真实训练使用 SentencePiece unigram tokenizer（byte fallback，中文字符不做
NFKC 改写）；byte tokenizer 只用于不依赖外部数据的 smoke test。

## 立即运行

项目固定 Python `>=3.11,<3.13`，`.python-version` 为 3.12。请始终通过 `uv run` 或
激活 `.venv` 运行；终端裸执行 `python` 若仍显示 3.14，只表示 base 环境在 PATH 前面，
不代表本项目虚拟环境异常。

```bash
uv sync
uv run miniscale doctor
uv run python -m unittest discover -s tests -v
uv run miniscale pipeline --device cpu --output artifacts/run
```

有可用 CUDA 时可将 `cpu` 改为 `cuda`。约 64M 参数的默认结构是 20 层、hidden size 512、
8 个 attention heads、2 个 KV heads、词表 8192、上下文 512。4GB 显存应从 batch size 1
开始；发生 OOM 时先把 sequence length 和 rollout group size 调低。

## 用真实数据运行完整链路

数据默认放在 `data/raw/minimind/`，并被 Git 忽略：

```text
pretrain/pretrain_t2t_mini.jsonl   预训练 text
sft/sft_t2t_mini.jsonl             多轮 conversations
preference/dpo.jsonl               chosen / rejected
rl/rlaif.jsonl                     需要 judge/reward model 的开放式 RLAIF 数据
agent/agent_rl_math.jsonl          带 gt 的数学工具任务
agent/agent_rl.jsonl               混合 Agent 任务
```

先训练 tokenizer。这一步只统计文本和训练分词器，不训练语言模型：

```bash
uv run miniscale train-tokenizer \
  --data data/raw/minimind/pretrain/pretrain_t2t_mini.jsonl \
  --output-prefix data/tokenizer/miniscale \
  --vocab-size 8192
```

然后逐阶段运行。下面的步数用于首次端到端实验，不代表最终收敛配置；每个阶段必须使用上个
阶段的权重和同一个 `miniscale.model`：

```bash
uv run miniscale pretrain --steps 10000 --batch-size 1 \
  --gradient-accumulation 16 --sequence-length 512 \
  --output artifacts/pretrain

uv run miniscale sft --steps 3000 --batch-size 1 \
  --gradient-accumulation 16 \
  --checkpoint artifacts/pretrain/pretrain.pt \
  --output artifacts/sft

uv run miniscale dpo --steps 1000 --batch-size 1 \
  --checkpoint artifacts/sft/sft.pt \
  --output artifacts/dpo

uv run miniscale grpo --steps 500 --batch-size 1 --group-size 4 \
  --checkpoint artifacts/dpo/dpo.pt \
  --output artifacts/grpo

uv run miniscale agent-rl --steps 500 --batch-size 1 --group-size 4 \
  --checkpoint artifacts/grpo/rl.pt \
  --output artifacts/agent-rl
```

当前 GRPO 默认使用 `agent_rl_math.jsonl`，因为其中的 `gt` 能形成确定的 verifier reward。
`rlaif.jsonl` 是开放式回答，不能拿“有没有数字”充当质量奖励；要使用它，应另接冻结的
reward model 或 LLM-as-a-judge。混合 `agent_rl.jsonl` 还包含当前 calculator sandbox 不支持的
工具，所以 Agent RL 也默认使用数学子集。

训练时终端会打印指标，并追加写入 `artifacts/<stage>/*_metrics.jsonl`。另开终端可以这样观察：

```bash
tail -f artifacts/pretrain/pretrain_metrics.jsonl
```

预训练主要看 `train_loss` 长期下降、`validation_loss` 同步下降，以及 `perplexity` 没有持续
反弹；单个 step 的抖动正常。DPO 看 loss 和 preference accuracy，GRPO/Agent RL 看 reward、
KL、clip fraction、success/tool-call rate。固定一组 held-out prompts，定期生成并保存结果，
比只盯 loss 更可信。

只想学习代码或验证安装时，继续使用秒级入口：

```bash
uv run python trainer/train_pretrain.py
uv run python trainer/train_sft.py
uv run python trainer/train_grpo.py
uv run python trainer/train_agent.py
```

输出在 `artifacts/`，该目录已被 Git 忽略。流水线清单会记录各阶段 loss、reward、耗时和
checkpoint 路径。

使用真实 SFT 或最后一个 Agent RL checkpoint 生成文本时必须传入训练时的 tokenizer：

```bash
uv run miniscale generate \
  --checkpoint artifacts/sft/sft.pt \
  --tokenizer data/tokenizer/miniscale.model \
  --prompt "你好，中国的首都是哪里？" \
  --temperature 0 \
  --max-new-tokens 100

# 也可以使用独立脚本；--raw 只打印模型回答
uv run python generate.py \
  --checkpoint artifacts/run/agent_rl.pt \
  --prompt "Use the calculator for 3*4." \
  --calculator \
  --temperature 0 \
  --raw
```

`pretrain.pt`、`sft.pt`、`dpo.pt`、`rl.pt` 和 `agent_rl.pt` 使用相同 checkpoint 格式，都可以生成；
评估 base `pretrain.pt` 的续写时可给 `generate` 增加 `--raw-prompt`。
但通常优先选择 `sft.pt` 或 `agent_rl.pt`。训练步数和数据量决定输出质量，文件能加载不等于
模型已经具备泛化能力。

## 建议学习顺序

1. `config.py`、`tokenizer.py`、`model.py`：亲手推导张量 shape，确认 causal test 为什么成立。
2. `data.py`、`training/pretrain.py`：理解流式读取、packing、shift label 和 next-token objective。
3. `training/sft.py`：检查 `-100` mask，确保 user/system/tool 文本不成为监督目标。
4. `training/dpo.py`：理解 policy/reference 对 chosen/rejected 的相对 log-probability。
5. `training/grpo.py`：先看 reward group normalization，再看 ratio clipping 和 reference KL。
6. `agent_env.py`、`training/agent_rl.py`：跟踪一次两轮 trajectory，确认 observation 进入上下文但
   action mask 为 0。
7. 修改一个组件并补测试，例如新增 `search` mock 工具、格式奖励或离线 eval；这才会逐渐变成
   你的工程，而不是 MiniMind 的复刻。

## 与工业训练栈的边界

这个小型实现保留了工业界最重要的语义：阶段化 checkpoint、assistant/action masking、
old/reference policy、可验证 reward、受限工具执行和端到端测试。但真正扩大训练规模前还需要：

- 为现有 SentencePiece/JSONL 数据加入去重、质量过滤、数据配比和评测污染检测；
- 加入 BF16、gradient checkpointing、FlashAttention、学习率调度、断点续训、FSDP/DeepSpeed；
- 将 rollout 与 learner 解耦，用 vLLM/SGLang 一类推理服务异步采样，并处理 policy weight 同步；
- 对工具执行使用进程/容器级 sandbox、超时、资源限额和审计，而不仅是当前的 AST 白名单；
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
