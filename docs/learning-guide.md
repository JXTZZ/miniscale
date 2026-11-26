# 从零学习 MiniScale

MiniScale 把语言模型训练拆成五个阶段。预训练学习预测下一个 token，SFT 教模型按对话格式回答，DPO 教模型表达偏好，GRPO 用可验证分数改进回答，Agent RL 继续学习在多轮过程中调用工具。沿着一道算术题，你可以从一行 JSONL 追到 loss、梯度和保存的权重。

你需要会运行 Python 和 shell 命令。前几节使用 CPU 和仓库自带的合成数据。每节先跑一个小实验，再读对应源码。以下命令都从仓库根目录运行，连续使用同一个 shell。

## 跑通第一次训练和推理

安装锁定的依赖，然后为本次实验建一个输出目录。

```bash
uv sync --locked
uv run --frozen python scripts/doctor.py
export OMP_NUM_THREADS=1
mkdir -p artifacts
smoke_dir=$(mktemp -d artifacts/learn-smoke.XXXXXX)

uv run --frozen python scripts/smoke_pipeline.py --device cpu --output "$smoke_dir"
uv run --frozen python eval_llm.py \
  --checkpoint "$smoke_dir/sft.pt" --prompt "Return only the result: 2+3" \
  --device cpu --temperature 0 --max-new-tokens 8
```

在推理结果里找 `response`、`generated_tokens` 和 `finish_reason`。这个小实验反复练习了 `2+3`，预期回答是 `5`。打开 `pipeline_manifest.json`，按顺序找出预训练、SFT、GRPO 和 Agent RL 的输出。Smoke 使用字节 tokenizer，命令省略 `--tokenizer` 即可读取这些权重。五阶段 JSONL 集成测试还覆盖了 DPO。 [Smoke 源码](../scripts/smoke_pipeline.py) · [五阶段集成测试](../tests/training/test_jsonl_training_chain.py)

## 从 token 找到 loss

Tokenizer 把文字编码成整数。设 batch 大小为 `B`、序列长度为 `T`、词表大小为 `V`。模型输入是 `[B, T]` 的 token ID，输出是 `[B, T, V]` 的 logits，也就是每个位置对词表中每个 token 的打分。

把 `[BOS, A, B, EOS]` 写在纸上。模型看到 `BOS` 时要预测 `A`，看到 `A` 时要预测 `B`，看到 `B` 时要预测 `EOS`。源码用 `logits[:, :-1]` 和 `labels[:, 1:]` 对齐这三个预测。运行测试后，找出改变后缀时保持相同的 logits，再找出调用 `backward()` 的位置。 [模型的 forward](../model/model_miniscale.py) · [测试源码](../tests/core/test_model_and_data.py)

```bash
uv run --frozen python -m unittest \
  tests.core.test_model_and_data.ModelAndDataTests.test_causality \
  tests.core.test_model_and_data.ModelAndDataTests.test_forward_and_backward -v
```

## 预训练两步并检查产物

下面的命令用默认 ByteLevel BPE tokenizer 和一层 decoder 做两次更新。每条输入记录是一个带 `text` 字段的 JSON 对象。

```bash
run_dir=$(mktemp -d artifacts/learn-train.XXXXXX)
uv run --frozen python - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
rows = [
    {"text": f"Arithmetic example {i}. Two plus three equals five. " * 20}
    for i in range(4)
]
(folder / "pretrain.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
)
PY

uv run --frozen python trainer/train_pretrain.py \
  --data "$run_dir/pretrain.jsonl" --output "$run_dir/pretrain" \
  --device cpu --steps 2 --num-hidden-layers 1 --sequence-length 64 \
  --gradient-accumulation 1 --shuffle-buffer-size 0 --validation-fraction 0 \
  --warmup-steps 1 --log-every 1 --save-every 1
cat "$run_dir/pretrain/pretrain_metrics.jsonl"
```

第二步的 `target_tokens_seen` 应为 `126`。每步送入 64 个 token，移位后有 63 个目标。阅读 `JsonlPretrainDataset` 时，找出一个块怎样把最后一个 token 留给下一块。接着在训练循环中定位 `backward()` 和 `optimizer.step()`。前者计算梯度，后者用梯度更新权重。 [预训练数据](../dataset/lm_dataset.py) · [预训练循环](../trainer/train_pretrain.py)

## 读懂后训练的四种信号

### SFT 只监督选中的回答

假设对话先回答 `5`，用户再问「乘以 3」，assistant 回答 `15`。把 SFT 的监督目标设为最后一轮，前一轮回答仍在输入上下文里。只有最后一轮的目标 token 计入 loss。代码用 `-100` 标记忽略的位置。打开 tokenizer 测试里的 `encode_sft` 调用，把 `response_only` 改成 `reasoning_and_response` 后检查哪些 token 会被监督。 [SFT tokenizer 测试](../tests/core/test_huggingface_tokenizer.py) · [SFT 数据索引](../dataset/sft_dataset.py)

### DPO 比较同一道题的两个答案

对同一个 prompt，给出 chosen 答案 `5` 和 rejected 答案 `4`。DPO 比较当前模型与冻结 reference 对两种答案的偏好差。当两个模型完全相同，偏好差为零，初始 sigmoid loss 为 `log(2)`。从测试里四个已知 log-prob 出发，先手算差值，再读 `dpo_loss`。 [DPO 公式测试](../tests/training/test_dpo_objective.py) · [DPO 目标](../trainer/preference_objective.py)

### GRPO 按同题候选的相对表现学习

对同一题生成三个回答，假设奖励为 `[0, 1, 2]`。减去组内均值后，三个优势的符号依次为负、零、正。代码还会按组内标准差缩放。读完这个步骤，再找两份 log-prob 的来源。采样时的策略用于求更新前后的比率，冻结 reference 用于计算 KL 约束。 [GRPO 目标](../trainer/rl/objective.py) · [共享策略训练循环](../trainer/rl/training.py)

### Agent RL 学习调用工具和使用返回值

模型先生成计算器调用，环境执行 `2+3`，把观测值 `5` 写回上下文，模型再生成最终回复。观测会影响后续生成，动作掩码只在模型自己生成的 token 上计算策略损失。用固定两轮回复的测试观察 `observation_ranges` 和 `action_mask`。 [轨迹实现](../trainer/rl/rollout.py) · [多轮工具测试](../tests/training/test_agent_rl.py)

```bash
uv run --frozen python -m unittest \
  tests.training.test_dpo_objective.DPOObjectiveTests.test_loss_matches_direct_formula \
  tests.training.test_grpo.GRPOTests.test_group_advantages_are_centered \
  tests.training.test_agent_rl.AgentRLTests.test_scripted_multiturn_rollout_masks_observation -v
```

## 区分权重交接和续训

`--checkpoint` 将已有权重作为新阶段起点，优化器从头建立。`--resume` 继续同一个实验，从文件恢复权重、优化器、scheduler、随机状态和数据进度。续训时 `--steps` 仍表示原计划的总步数。

用回归测试验证这个区别。测试分别执行连续训练和断点恢复，然后比较最终权重。读完测试后，去 `trainer/runtime/run.py` 找安全更新边界，去 `trainer/checkpoint.py` 找保存与恢复。 [恢复契约](checkpointing.md) · [安全更新边界](../trainer/runtime/run.py)

```bash
uv run --frozen python -m unittest \
  tests.training.test_pretrain_resume.PretrainResumeTests.test_interrupt_after_optimizer_uses_last_complete_checkpoint -v
```

## 试用可选训练功能

先用 LoRA 做两个 SFT 更新。LoRA 冻结原模型，只在每层 attention 的 query 和 value 投影上学习低秩增量。`--lora-rank 4` 指定秩，运行清单的 `trainable_parameters` 可和模型总参数量对照。完整训练 checkpoint 保存适配器以支持续训，推理和阶段交接会加载合并后的权重。 [LoRA 实现](../model/lora.py)

```bash
uv run --frozen python trainer/train_full_sft.py \
  --data "$smoke_dir/sft.jsonl" --output "$run_dir/lora" \
  --checkpoint "$run_dir/pretrain/final.pt" --lora-rank 4 \
  --device cpu --steps 2 --gradient-accumulation 1 --min-context-tokens 8 \
  --validation-fraction 0 --generation-every 0 --log-every 1
```

再用这个 checkpoint 试一次 RL。CISPO 把截断后的重要性比率当作固定权重，乘到当前策略的 log-prob 上。`--batch-rollouts` 一次采样同一道题的多个候选。`--kv-cache` 让生成复用已处理 token 的 Key 和 Value。下面的一步训练用来核对功能和日志，奖励会随小模型的采样结果变化。 [CISPO 目标](../trainer/rl/objective.py) · [批量 rollout](../trainer/rl/rollout.py)

```bash
printf '%s\n' '{"conversations":[{"role":"user","content":"2+3?"}],"gt":["5"]}' > "$run_dir/rl.jsonl"
uv run --frozen python trainer/train_grpo.py \
  --data "$run_dir/rl.jsonl" --output "$run_dir/cispo" \
  --checkpoint "$run_dir/lora/sft.pt" --loss-type cispo --batch-rollouts --kv-cache \
  --device cpu --steps 1 --group-size 2 --max-new-tokens 8 --validation-fraction 0
```

打开 `$run_dir/cispo/grpo_run.json`，查看已选的目标函数和采样配置。对比采样或目标函数时，每次只换一个开关并使用新的输出目录。希望检查 KV cache 时，先运行缓存等价测试，再读生成循环。 [缓存等价测试](../tests/core/test_model_and_data.py) · [生成循环](../model/model_miniscale.py)

## 后续阅读

按 [工程结构](architecture.md) 找模块职责，按 [数据格式](../dataset/dataset.md) 换成自己的语料。模型基础和训练循环可对照 PyTorch 教程，tokenizer、数据和微调的概念可继续读 Hugging Face LLM Course。 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) · [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
