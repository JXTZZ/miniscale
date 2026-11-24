# MiniScale 学习指南

从 [`scripts/smoke_pipeline.py`](../scripts/smoke_pipeline.py) 的 `run_training_pipeline` 开始。先在 CPU 上跑通训练和推理，再沿着一条数据看它如何变成 token、进入 loss 并更新模型。熟悉这条路径后，用小测试观察 SFT、DPO、GRPO 和 Agent RL 各自的训练信号。

## 跑一次训练和推理

在仓库根目录运行下面的命令。每次实验使用新输出目录，方便保留前一轮的产物。

```bash
uv sync --locked
uv run --frozen python scripts/doctor.py

mkdir -p artifacts
smoke_dir=$(mktemp -d artifacts/learn-smoke.XXXXXX)
OMP_NUM_THREADS=1 uv run --frozen python scripts/smoke_pipeline.py \
  --device cpu --output "$smoke_dir"
uv run --frozen python eval_llm.py \
  --checkpoint "$smoke_dir/sft.pt" --prompt "2+3?" \
  --device cpu --temperature 0 --max-new-tokens 8
```

打开 `"$smoke_dir/pipeline_manifest.json"`，对照代码找出四个阶段各自的输入和产物。Smoke 使用内存样例和 `ByteTokenizer`，用来验证训练、保存和加载。观察推理报告中的 `response`、`generated_tokens` 和 `finish_reason`。正式 JSONL 链路包含 DPO，测试会在阶段交接时逐次加载上游权重。 [五阶段测试](../tests/training/test_jsonl_training_chain.py)

```bash
uv run --frozen python -m unittest \
  tests.training.test_jsonl_training_chain.JsonlTrainingChainTests.test_all_training_stages_read_jsonl -v
```

## 跟踪输入到 loss 的路径

打开 `MiniScaleForCausalLM.forward`，从输入和输出的 shape 开始。设 `B` 为批大小、`T` 为序列长度、`H` 为隐藏维度、`V` 为词表大小。Token ID 的形状是 `[B, T]`，embedding 变成 `[B, T, H]`，decoder 层保留这个形状，最后的词表投影输出 `[B, T, V]`。先用 `[BOS, A, B, EOS]` 手算一遍，三个预测目标依次是 `A`、`B`、`EOS`。代码里的 `logits[:, :-1]` 和 `labels[:, 1:]` 实现了这个对齐。 [模型实现](../model/model_miniscale.py)

先跑下面两条断言。读 `test_causality` 时，观察改变输入后缀会影响哪些位置的 logits。读 `test_forward_and_backward` 时，找出构造 batch、计算 loss 和反向传播的三行代码。接着进入 `CausalSelfAttention.forward`，跟踪 RoPE 应用到哪些张量，以及 GQA 在哪里把少量 KV heads 对齐到 query heads。 [模型测试](../tests/core/test_model_and_data.py)

```bash
uv run --frozen python -m unittest \
  tests.core.test_model_and_data.ModelAndDataTests.test_causality \
  tests.core.test_model_and_data.ModelAndDataTests.test_forward_and_backward -v
```

## 用两步预训练读懂正式入口

从 `train_pretrain.py` 的 `main` 跟到 `run_pretrain_jsonl`。第一遍只跟 `JsonlPretrainDataset`、`model(**batch)`、`backward()` 和 `optimizer.step()`。观察数据集如何为每篇文档加 BOS 和 EOS，把 token 拼成等长块，并把一个边界 token 留给下一块作上下文。之后再看梯度累计、验证采样和 checkpoint 保存。 [预训练入口](../trainer/train_pretrain.py) · [数据读取](../dataset/lm_dataset.py)

在同一个 shell 里运行下面的实验。它生成几行本地语料，用一层模型做两次更新，并保留每步的指标。

```bash
mkdir -p artifacts
run_dir=$(mktemp -d artifacts/learn-pretrain.XXXXXX)
uv run --frozen python - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

corpus = Path(sys.argv[1]) / "corpus.jsonl"
with corpus.open("w", encoding="utf-8") as stream:
    for index in range(4):
        text = f"Document {index}. A language model predicts the next token. " * 16
        stream.write(json.dumps({"text": text}) + "\n")
PY

OMP_NUM_THREADS=1 uv run --frozen python trainer/train_pretrain.py \
  --data "$run_dir/corpus.jsonl" --output "$run_dir/train" \
  --device cpu --seed 42 --steps 2 --num-hidden-layers 1 \
  --batch-size 1 --sequence-length 64 --gradient-accumulation 1 \
  --shuffle-buffer-size 0 --validation-fraction 0 --warmup-steps 1 \
  --generation-every 0 --log-every 1 --save-every 1

cat "$run_dir/train/pretrain_metrics.jsonl"
```

检查两条记录的 `step`、`learning_rate` 和 `target_tokens_seen`。每步送入 64 个 token，其中 63 个位置提供 next-token 目标，第二步结束时监督目标累计为 126。把这三个字段和循环里的更新语句对应起来。再打开 `pretrain_run.json` 看解析后的参数和数据指纹，并对照 `checkpoints/step_00000001.pt` 与 `final.pt` 的保存位置。 [预训练细节](pretraining.md)

## 用小测试读懂 SFT 到 Agent RL

按下表顺序阅读。每次先看测试用的样本，预测断言结果，再进入实现检查预测。

| 阶段 | 从哪里读 | 动手检查 |
| --- | --- | --- |
| SFT | [`HuggingFaceTokenizer.encode_sft`](../model/tokenizer.py) → [`IndexedJsonlSFTDataset`](../dataset/lm_dataset.py) → [`run_sft_jsonl`](../trainer/train_full_sft.py) | 把 `labels` 中非 `-100` 的位置解码，检查哪些 assistant token 被监督。 |
| DPO | [`encode_preference_pair`](../dataset/preference_dataset.py) → [`completion_log_probability` 和 `dpo_loss`](../trainer/train_dpo.py) | 用四个固定 log-prob 手算 loss。把 policy 设为与 reference 相同，预期 margin 为 0、loss 为 `log(2)`。 |
| GRPO | [`run_grpo`](../trainer/train_grpo.py) → [`collect_rollouts`](../trainer/rollout_engine.py) → [`normalize_group_rewards` 和 `grpo_objective`](../trainer/rl_utils.py) | 给同一题的三个回答设定奖励 `[0, 1, 2]`，推导组内优势的符号。找出用于比率裁剪的 `old_log_probs` 何时记录，以及冻结 reference 如何提供 KL 基线。 |
| Agent RL | [`rollout_agent`](../trainer/rollout_engine.py) → [`CalculatorEnv`](../trainer/agent_env.py) → [`run_agent_grpo`](../trainer/train_agent.py) | 跟踪一次 `2+3` 的工具调用与返回值 `5`。检查工具观测如何进入后续上下文，以及 `observation_ranges` 对应的 action mask 为什么全为 0。 |

用下面的代码观察 SFT 的监督目标。两轮消息都传给 tokenizer，`target_assistant_index=-1` 选择最后一轮 assistant，`response_only` 选择回复部分。输出中可以看到最后一轮回复 `15` 和模板的结束标记。

```bash
uv run --frozen python - <<'PY'
from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer

tokenizer = load_tokenizer(DEFAULT_TOKENIZER_PATH)
messages = [
    {"role": "user", "content": "What is 2 + 3?"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "Multiply that result by 3."},
    {"role": "assistant", "content": "15", "reasoning_content": "Multiply five by three."},
]
_, labels = tokenizer.encode_sft(
    messages, target_assistant_index=-1, target_mode="response_only"
)
targets = [label for label in labels if label != -100]
print(repr(tokenizer.decode(targets, skip_special_tokens=False)))
PY
```

把 `target_mode` 改为 `reasoning_and_response` 再运行，看看监督文本增加了什么。用户问题和历史回复仍作为上下文参与预测，`labels=-100` 只控制对应位置是否直接计入 loss。 [SFT mask 断言](../tests/core/test_huggingface_tokenizer.py)

读偏好训练和 RL 时，运行下列小测试。Agent 测试用固定的两轮回复触发计算器，可以逐个检查 observation token 的边界。

```bash
uv run --frozen python -m unittest \
  tests.training.test_dpo_objective.DPOObjectiveTests.test_loss_matches_direct_formula \
  tests.training.test_grpo.GRPOTests.test_group_advantages_are_centered \
  tests.training.test_agent_rl.AgentRLTests.test_scripted_multiturn_rollout_masks_observation -v
```

## 验证一次恢复和一次改动

`--checkpoint` 为 SFT、DPO、GRPO 或 Agent RL 的新训练提供初始权重。`--resume` 用于继续同一阶段的完整训练，恢复优化器、scheduler、随机状态和数据进度。恢复时，`--steps` 继续填写原计划的总步数。读 `read_training_checkpoint` 与 `restore_training_checkpoint` 时，沿调用者找到恢复前的签名校验。 [Checkpoint 契约](checkpointing.md) · [保存和恢复实现](../trainer/checkpoint.py)

```bash
uv run --frozen python -m unittest \
  tests.training.test_pretrain_resume.PretrainResumeTests.test_periodic_checkpoint_retention_and_resume -v
```

最后做一次只改一个变量的小实验。例如，在两步预训练命令里把 `--gradient-accumulation` 改为 `2`，用新的 `--output` 运行。先预测第二步的累计监督 token 数，再用 JSONL 核对，并解释为什么 optimizer 仍更新两次。阅读到这里，可以用 [阶段数据格式](../dataset/dataset.md) 和 [README 的正式训练命令](../README.md#用真实数据运行完整链路) 换成自己的数据。
