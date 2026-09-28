# rebuild 阅读与动手顺序

先读 [本次检查报告](rebuild-review.md)，再按下面的顺序学习。所有训练命令从 `rebuild/` 执行。
`rebuild` 是目录，`learn/rebuild-from-scratch` 是 Git 分支；进入目录不会切换分支。

## 0. 确认运行的是哪份代码

根目录和 rebuild 都提供名为 `miniscale` 的 Python 包，因此先确认导入路径：

```bash
cd /home/lotusy/hy/miniscale/rebuild
PYTHONPATH=src ../.venv/bin/python -c 'import miniscale; print(miniscale.__file__)'
PYTHONPATH=src ../.venv/bin/python -m miniscale doctor
```

第一行应输出含 `/rebuild/src/miniscale/` 的路径。这两条命令复用已有根目录虚拟环境。
如果想让 rebuild 拥有独立环境，运行 `uv sync`，之后改用 `uv run python` 和 `uv run miniscale`。

先跑一次现有的小型集成示例：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  ../.venv/bin/python -m miniscale pipeline --device cpu --output artifacts/learning-smoke
```

打开 `artifacts/learning-smoke/pipeline_manifest.json`，对应阅读
[`pipeline.py`](../src/miniscale/pipeline.py)。这里的 SFT 刻意重复学习两个例子，验证阶段能衔接，
不能用它的回答判断正式模型质量。再次运行时换一个输出目录。

## 1. 先理解一批数据怎么变成 loss

| 顺序 | 文件 | 阅读时要回答的问题 |
| --- | --- | --- |
| 1 | [config.py](../src/miniscale/config.py) | 词表、宽度、层数、上下文分别控制什么？ |
| 2 | [tokenizer.py](../src/miniscale/tokenizer.py) | 文本如何变成整数？BOS/EOS、chat template、`-100` 分别起什么作用？ |
| 3 | [data/pretrain.py](../src/miniscale/data/pretrain.py) | 多篇文档如何 packing？相邻 block 为什么重叠一个 token？ |
| 4 | [data/batches.py](../src/miniscale/data/batches.py) | input、label、attention mask 的 shape 和 padding 是什么？ |
| 5 | [model.py](../src/miniscale/model.py) | `[B,T]` 如何变为 `[B,T,V]`？loss 在哪里做 next-token shift？ |

模型先看 `MiniScaleForCausalLM.forward`，再沿调用读 `DecoderLayer`、attention、RoPE、SwiGLU、RMSNorm。
第一次阅读先记录张量 shape，第二遍再推公式。

动手练习：把一行文本编码后打印前 20 个 token；手写一个长度为 5 的序列，画出
`logits[:, :-1]` 和 `labels[:, 1:]` 的对应关系。不要在 dataset 和模型中各 shift 一次。

## 2. 跟完一次预训练更新

按顺序读：

1. [pretrain/config.py](../src/miniscale/training/pretrain/config.py)：参数与默认值。
2. [pretrain/runner.py](../src/miniscale/training/pretrain/runner.py) 的 `run_pretrain_jsonl`：真实训练入口。
3. [core/runtime.py](../src/miniscale/training/core/runtime.py)：设备、AdamW 分组、学习率调度。
4. [pretrain/eval.py](../src/miniscale/training/pretrain/eval.py)：固定生成探针；验证 loss 使用共享 `evaluate_lm`。
5. [pretrain/resume.py](../src/miniscale/training/pretrain/resume.py) 和 [core/checkpoint.py](../src/miniscale/training/core/checkpoint.py)：恢复时要保存什么。

在纸上标出：取 batch → autocast → forward → loss/梯度累计 → backward → clip → optimizer.step → scheduler.step → zero_grad。
注意 `steps` 是优化器更新次数。batch=1、梯度累计=16、序列长=768 时，每次更新的 next-token 目标数是
`1 × 16 × (768 - 1) = 12,272`。

推荐配套测试：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src ../.venv/bin/python -m unittest \
  tests.training.test_pretrain tests.training.test_pretrain_resume -v
```

先复现“连续训练”和“中途保存再恢复”的权重一致，再尝试修改 batch 或学习率。

## 3. 再学习 SFT，重点看监督范围

阅读 [data/sft.py](../src/miniscale/data/sft.py)、`HuggingFaceTokenizer.encode_sft`、
[sft/config.py](../src/miniscale/training/sft/config.py)、[sft/runner.py](../src/miniscale/training/sft/runner.py)。

重点弄清：

- 一个多轮 conversation 会展开为多个 assistant-turn 样本。
- prompt 和历史轮次的 label 为 `-100`；当前 assistant 回复才计算监督 loss。
- `reasoning_and_response` 与 `response_only` 如何改变监督范围。
- 截断如何保留上下文和回复，为什么把长答案截掉会改变学习内容。

动手打印一个普通对话和一个 tool call 样本的 token/label 对照；检查 assistant 的工具调用被监督，
工具 observation 没被误当成 assistant 输出。随后读 [sft/eval.py](../src/miniscale/training/sft/eval.py)：
较低的训练 loss 并不自动意味着更少复读或更好的对话。

## 4. 最后看偏好优化和强化学习

1. DPO：`dpo/objective.py` → `data/preference.py` → `dpo/runner.py`。
   先弄清 chosen/rejected、共享 prompt、冻结 reference 和 beta。
2. GRPO：`grpo/rollout.py` → `grpo/objective.py` → `grpo/runner.py` → `grpo/eval.py`。
   先看组内 reward/advantage，再看 old policy、clip 和 reference KL。
3. Agent RL：根模块 `agent_env.py` → `agent_rl/rollout.py` → `agent_rl/runner.py`。
   `agent_rl/env.py` 目前转发根环境实现。重点看多轮工具执行和 action mask：环境 observation 不参与策略 loss。

GRPO 默认读取 `agent_rl_math.jsonl` 的可验证答案。`rlaif.jsonl` 只有开放式回答，
要另接 reward model 或 judge；下载文件不等于当前训练器已经支持它的奖励计算。

## 5. 每次只做一个可验证改动

在 `experiments/` 记录假设、基线、候选、seed、模型/数据/tokenizer 身份和结果：

- 质量改动：固定训练 token 预算与验证集，比较 loss、生成质量或成功率。
- 性能改动：固定模型、batch、上下文、精度和数据，比较 tok/s、峰值显存和完整耗时。
- 代码重构：要求语义测试与恢复测试通过，并对照固定 seed 的短跑权重。

第一轮适合做“读懂一个完整预训练 step + 改一项日志 + 跑恢复测试”。
不要同时调整模型层数、tokenizer、数据和训练参数，否则无法判断差异来源。
`recipes/` 和 `experiments/` 当前只有说明，recipe 加载器和统一实验对照工具仍待实现。
