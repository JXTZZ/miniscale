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
JSONL 文本 -> MiniMind ByteLevel BPE -> Pretrain (next-token loss)
            -> SFT (仅 assistant token loss)
            -> DPO (chosen/rejected + frozen reference)
            -> RLVR/GRPO (可验证数学奖励 + group advantage + clip + reference KL)
            -> Agent GRPO (多轮 tool call / observation / action-only loss)
            -> 各阶段 checkpoint + JSONL metrics
```

模型部分原生 PyTorch 实现了 RMSNorm、RoPE、GQA、SDPA causal attention、SwiGLU、
tied embeddings。真实训练默认使用 MiniMind 已训练好的 6400 词表 Hugging Face tokenizer，
并直接复用其 chat template、思考标签和工具调用格式；byte tokenizer 只用于 smoke test，
本仓库训练 SentencePiece 的入口则保留为对照实验。

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
8 个 attention heads、2 个 KV heads、词表 6400、上下文 512。4GB 显存应从 batch size 1
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

MiniMind tokenizer 已放在 `data/tokenizer/minimind/`，包含 `tokenizer.json` 和
`tokenizer_config.json`，不需要重新训练。可以先验证它：

```bash
uv run python -c "from miniscale.tokenizer import load_tokenizer; t=load_tokenizer('data/tokenizer/minimind'); print(type(t).__name__, t.vocab_size)"
```

预期输出 `HuggingFaceTokenizer 6400`。如果需要做 tokenizer 消融实验，仍可使用
`miniscale train-tokenizer` 训练独立 SentencePiece 模型，但它产生的 checkpoint 与默认
MiniMind tokenizer checkpoint 不兼容。

输入任意文本查看 token ID、子词、压缩率和解码结果：

```bash
uv run miniscale tokenize \
  --text "你好，中国的首都是北京。MiniScale正在学习Agent RL！" \
  --add-bos \
  --add-eos
```

`round_trip` 应为 `true`；`characters_per_token` 越高表示这段文本压缩得越好，但应在固定的
中英/代码/数学测试集上比较，而不是只看一个句子。ByteLevel BPE 的 `tokens` 可能显示成
`ä½łå¥½` 一类内部字节符号，这是正常表示，判断正确性应看 `decoded` 和 `round_trip`。

然后逐阶段运行。下面的步数用于首次端到端实验，不代表最终收敛配置；每个阶段必须使用上个
阶段的权重和同一个 `data/tokenizer/minimind/` 目录：

```bash
uv run miniscale pretrain --steps 10000 --batch-size 1 \
  --gradient-accumulation 16 --sequence-length 768 \
  --learning-rate 3e-4 --min-learning-rate 3e-5 \
  --warmup-steps 200 --save-every 500 --keep-last 3 \
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

预训练前 200 step 线性 warmup 到 `3e-4`，之后 cosine decay 到 `3e-5`。每 500 step
在 `artifacts/pretrain/checkpoints/` 保存完整训练状态，并只保留最近 3 个周期 checkpoint；
完整状态包含模型、AdamW、scheduler、step、token 计数和随机数状态。按 `Ctrl+C` 时还会写入
`emergency_step_XXXXXXXX.pt` 后再退出。完整 checkpoint 通常约 700–800MB，请预留磁盘空间。

从周期或 emergency checkpoint 继续时，`--steps` 仍表示原计划的总步数，其他影响训练轨迹的
参数必须与原命令一致：

```bash
uv run miniscale pretrain --steps 10000 --batch-size 1 \
  --gradient-accumulation 16 --sequence-length 768 \
  --learning-rate 3e-4 --min-learning-rate 3e-5 \
  --warmup-steps 200 --save-every 500 --keep-last 3 \
  --resume artifacts/pretrain/checkpoints/step_00000500.pt \
  --output artifacts/pretrain
```

恢复时数据流会跳过 checkpoint 已消费的 micro-batches，metrics 文件中晚于恢复 step 的旧记录
会被移除，避免重复 step。旧版仅包含模型权重的 `pretrain.pt` 没有 optimizer/scheduler 状态，
不能用 `--resume`；但仍可用于生成或作为后续 SFT 的初始权重。

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
  --tokenizer data/tokenizer/minimind \
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

- 为现有 BPE/JSONL 数据加入去重、质量过滤、数据配比和评测污染检测；
- 加入 BF16、gradient checkpointing、FlashAttention、FSDP/DeepSpeed；
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
