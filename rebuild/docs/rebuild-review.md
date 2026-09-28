# rebuild 检查报告（2026-09-28）

## 结论

本次未发现重构引入的训练计算差异。根目录项目与 rebuild 在相同 seed 的 CPU 短跑中，
24 份 checkpoint 的模型权重逐张量完全相同。因此，在模型、tokenizer、数据、seed、训练预算和精度相同时，
预期训练效果应接近。没有运行 4090 上的完整训练，不能把短跑一致性当作完整质量或速度保证。

当前默认模型实测为 **63,590,912 参数（63.6M）**，20 层、hidden=512、FFN=1536、Q heads=8、KV heads=2、词表=6400。
这只是继承的现有默认配置，尚未落实此前讨论的约 123M 方案；模型大小应在独立实验中决定。
123M 表示 1.23 亿，1.27 亿是 127M，两者不是同一个精确数值。

## 与根目录实现的核对

比较基线：根目录/main `6b485e4`；重构实现 `a2e8630`。

| 项目 | 结果 |
| --- | --- |
| `model.py`、`config.py`、`tokenizer.py` | 文件内容完全一致 |
| `pipeline.py` | 文件内容完全一致 |
| checkpoint、设备/优化器/学习率公共代码 | 实现保留；部分导入路径调整 |
| 231 个顶层函数/类的语法树对照 | 230 个找到完全相同的实现；1 个 packing 类实现调整 |
| 根目录完整测试 | 111 项通过 |
| rebuild 完整测试 | 113 项通过 |
| CPU、seed=42、单线程的 smoke 和 JSONL 五阶段对照 | 24 份 checkpoint 模型权重完全一致 |
| 真实 mini 数据 + 本地 MiniMind tokenizer | 各取前 64 行，微型模型预训练 2 step → 加载 checkpoint → SFT 2 step，均完成且 loss 有限 |

唯一的类实现变化是 `JsonlPretrainDataset._iter_packed_examples`：旧版每产生一个 block 就删除列表头部，
新版用 offset 读取并在每篇文档结束后一次删除已消费 token。block 顺序、单 token 重叠和尾部丢弃规则保留。
该修改减少长文档的重复数据移动，但本次没有测量它在真实 4090 训练中的速度收益。

检查产物保存在 `artifacts/review/`，其中 `equivalence.json` 记录各 checkpoint 的对照结果，
`real-data-smoke/result.json` 记录真实数据短跑结果。这些产物被 Git 忽略。

测试复现（从仓库根目录执行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -q
```

进入 rebuild 后执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  ../.venv/bin/python -m unittest discover -s tests -q
```

## 两种 pipeline 要分清

`miniscale pipeline` 使用 ByteTokenizer 和微型模型，执行：

```text
少量内存文本 → pretrain → 两条对话 SFT → GRPO → Agent RL
```

它用于检查阶段能衔接，未包含 DPO。正式 JSONL 阶段分别通过 `pretrain`、`sft`、`dpo`、`grpo`、`agent-rl` 命令启动，
上一阶段 checkpoint 显式传给下一阶段。JSONL 集成测试覆盖这五个阶段。

与 MiniMind 官方训练效果不能直接画等号：MiniScale 复用其数据和 tokenizer，但模型几何、数据切分/packing、
训练步数、有效 batch、优化器和后训练实现由本项目定义。只有完整对齐配方并在同一评估集实测，才能比较质量。

## 重构前后路径映射

以下路径相对于各自的 `src/miniscale/`：

| 根目录项目 | rebuild |
| --- | --- |
| `data/__init__.py` 中的数据实现 | `data/jsonl.py`、`batches.py`、`pretrain.py`、`sft_smoke.py` |
| `training/stages/pretrain.py` | `training/pretrain/{runner,eval,resume,manifest}.py` |
| `training/configs/pretrain.py` | `training/pretrain/config.py` |
| `training/stages/sft.py` 与对应 configs/evaluators | `training/sft/` |
| `training/stages/dpo.py` 与对应 configs/objectives/evaluators | `training/dpo/` |
| `training/stages/grpo.py` 与对应 objective | `training/grpo/`，rollout/eval 已拆出 |
| `training/stages/agent_rl.py` | `training/agent_rl/`，rollout/eval 已拆出 |
| `training/configs/rl.py` | 共用定义在 `training/core/rl_config.py`，阶段 config 提供入口 |

rebuild 下的 `configs/`、`stages/`、`objectives/`、`evaluators/` 仅作旧导入转发。
学习新实现从各阶段目录进入。`agent_rl/env.py` 目前仍转发根 `agent_env.py`，环境实现只维护一份。

## 数据下载与完整性

来源：[ModelScope / gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。
[MiniMind 官方说明](https://github.com/jingyaogong/minimind#readme) 当前推荐 mini 预训练和 mini SFT 组合用于快速复现。
本次按数据仓库列表确认文件，使用各文件的 revision 下载，并逐个对照上游 SHA-256；全部通过。

| 相对于 `data/raw/minimind/` 的路径 | 精确字节数 | 用途 |
| --- | ---: | --- |
| `pretrain/pretrain_t2t_mini.jsonl` | 1,241,043,656 | 预训练 |
| `sft/sft_t2t_mini.jsonl` | 1,739,201,170 | 监督微调 |
| `preference/dpo.jsonl` | 53,653,322 | 偏好对，共 17,166 行 |
| `agent/agent_rl_math.jsonl` | 18,372,683 | 数学 GRPO 与计算器 Agent，共 20,000 行 |
| `agent/agent_rl.jsonl` | 82,036,930 | 混合 Agent 任务，共 39,988 行 |
| `rl/rlaif.jsonl` | 23,754,740 | 开放式回答，共 19,502 行；另需奖励模型或 judge |

后四份文件也已逐行解析，均为 JSON 对象。混合 Agent 数据包含当前计算器环境不支持的工具，
现有 Agent 训练仍使用数学子集。不要把下载齐全误解为所有任务都有可执行环境。

[`scripts/download_minimind.py`](../scripts/download_minimind.py) 支持中断续传和已有文件校验。
`data/raw/minimind/download_manifest.json` 保存完整下载 URL、revision、字节数与 SHA-256。
大数据不提交到 Git，后续机器可以用同一脚本下载；上游更新时应核对 manifest 的版本和 hash。

### 预训练全量审计

使用词表 6400 的本地 tokenizer、sequence length=768、验证比例=0.005：

- 1,270,238 篇有效文档；无空行、JSON 错误或空文本。
- 191 行精确重复，当前预训练读取器保留它们；内容 hash 切分让相同文本落入同一侧。
- 总 token stream 为 332,495,324（含每篇的 BOS/EOS）。
- 训练集 packing 后有 431,398 个 block，330,882,266 个 next-token 目标。
- 验证集有 2,102 个 block；没有执行近重复或 benchmark 污染检查。

README 的 10,000 steps × batch 1 × 累计 16 × 767 = **122,720,000 个训练目标 token**，
约相当于上述训练 stream 的 **37.1%**。它不是“一遍全部 mini 语料”；同样配置下约 26,963 次更新才覆盖一遍的量级。
这只是数据覆盖估算，不是收敛或最优训练步数的保证。

完整报告：`artifacts/review/pretrain-data-audit.json`。

### SFT 全量结构与长度对照

本次逐行解析了 905,718 个 conversation，发现 1,249,662 个有内容、reasoning 或 tool call 的 assistant 目标。
这个总数尚未扣除重复和验证集。用 seed=42 做全局 reservoir sampling，抽取 2,000 个 assistant-turn 样本，
通过实际 tokenizer 和 `truncate_sft_example` 比较两种长度：

| max length | 输入发生截断 | 目标回复发生截断 | 监督 token 保留率 | 无监督 token 样本 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 36.65% | 21.90% | 92.93% | 0 |
| 768 | 3.60% | 0.95% | 99.74% | 0 |

因此，使用这批 mini SFT 数据且预训练 checkpoint 上下文至少为 768 时，建议先用显式 `--max-length 768` 建立基线。
这是当前数据样本的长度证据，尚未通过完整训练比较两种长度的最终质量和速度。
README 的 3,000 steps × batch 1 × 累计 16 只处理约 48,000 个 assistant-turn 样本，
相对于去重和验证切分前的总数约为 3.8%；示例步数不能代表完整 SFT 一轮或已收敛。

完整质量扫描还发现：

- 无无效 conversation；46,862 条 conversation 为精确重复。正式 SFT 默认 `deduplicate_exact=True`，会跳过这些重复。
- 29 个 assistant 消息没有可监督目标，读取器会跳过。`content` 为空但含 tool call 或 reasoning 的消息不等于无效样本。
- 308,390 个 assistant 消息带 reasoning，77,784 个带 tool call，说明 assistant mask 和工具模板确实影响这批数据。
- 78,122 个 assistant 目标（约 6.25%）被现有严重重复规则标记。这是启发式筛查，不是人工确认的低质量率；
  代码、列表和格式化文本可能误报，应先抽查再决定过滤策略。
- 审计结构计数为保留重复时的结果；实际训练按默认去重后的样本数会更少。本轮保留了所有原始下载文件。

完整报告：`artifacts/review/sft-data-audit.json`；长度对照：`artifacts/review/sft-length-comparison.json`。
真实前 64 行的训练 smoke 只检查数据链路，不评估泛化质量。

可在 rebuild 目录复现审计：

```bash
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 PYTHONPATH=src ../.venv/bin/python -m miniscale \
  audit-pretrain-data --sequence-length 768 --output artifacts/review/pretrain-data-audit.json
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=4 PYTHONPATH=src ../.venv/bin/python -m miniscale \
  audit-sft-data --max-length 768 --sample-size 2000 --output artifacts/review/sft-data-audit.json
```

SFT 命令会扫描全量文本并计算重复特征，运行明显慢于单纯 JSON 校验；这不代表训练时每步都会执行该审计。


## 尚未完成的部分与下一步

- `recipes/`、`experiments/` 目前只有说明；可加载 recipe 和统一实验对照报告仍未实现。
- 较长的 runner 仍可逐步整理。薄兼容层、CLI、训练循环之间的职责已明确，但本次不再扩大结构改动。
- 没有测过这套配置在 4090 上的峰值显存、BF16 长跑稳定性或最终对话质量。
- 下一步先按 [学习指南](learning-guide.md) 跑通一个完整 step，再固定一份正式配方，记录质量与吞吐基线。
- 学习和合并流程见 [Git 指南](git-workflow.md)。当前根目录代码仍保留，合并该分支不会自动替换旧项目。
