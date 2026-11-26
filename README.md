# MiniScale

MiniScale 是一个从零实现、可测试的“小模型到 Agent RL”学习项目。项目参考了 MiniMind 的训练路线
和 MokioMind 的手写学习方式。目录重构采用 MiniMind 的平铺布局，将 MiniScale 原有的模型、数据管线、
训练目标和工具环境迁入 `model/`、`dataset/`、`trainer/` 和 `scripts/`。 [MiniMind](https://github.com/jingyaogong/minimind) · [MokioMind](https://github.com/Wood-Q/MokioMind)

仓库同时保留秒级 smoke pipeline，并提供约 64M 参数模型的真实 JSONL 训练入口。真实入口
直接读取本地数据，不会把十几 GB 数据一次性装入内存；训练步数必须显式指定，避免误启动
长任务。

## MiniScale 的训练特性

- **可复现的预训练与续训**。以 `--steps` 指定训练预算，流式打包 JSONL，按内容稳定切分验证集。完整 checkpoint 保存数据与 tokenizer 指纹、训练进度、优化器、scheduler 和随机状态。 [预训练](docs/pretraining.md) · [恢复契约](docs/checkpointing.md)
- **SFT 样本选择与质量评估**。按 assistant turn 选择监督目标，可用质量策略生成派生数据，使用固定生成套件评估回复并选择 checkpoint。 [SFT](docs/sft.md)
- **偏好训练与可验证奖励**。DPO 校验共享 prompt 并保留冻结 reference，GRPO 用数学答案计算奖励，Agent RL 用 action mask 从优化目标中排除计算器观测。 [DPO](docs/dpo.md) · [GRPO](docs/grpo.md) · [Agent RL](docs/agent_rl.md)
- **可选训练扩展**。SFT 支持 LoRA，GRPO 与 Agent RL 可选择 CISPO，单轮 GRPO 可批量采样，推理与 RL 可启用 KV cache。 [动手实验](docs/learning-guide.md#试用可选训练功能)
- **本地指标与 W&B**。五个阶段保存本地 JSONL 指标，并可选用 W&B。训练中断后可继续使用原 run ID，持久化的待传队列会在重连后补传指标和生成样例。 [W&B 实现](trainer/runtime/tracking.py)

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

第一次阅读项目可以从 [MiniScale 学习指南](docs/learning-guide.md) 开始。指南从 CPU smoke
出发，用小实验串起模型、数据、五阶段训练和 checkpoint 恢复。

项目固定 Python `>=3.11,<3.13`，`.python-version` 为 3.12。请始终通过 `uv run` 或
激活 `.venv` 运行；终端裸执行 `python` 若仍显示 3.14，只表示 base 环境在 PATH 前面，
不代表本项目虚拟环境异常。

```bash
uv sync --locked
uv run --frozen python scripts/doctor.py
uv run --frozen python -m unittest discover -s tests -v
uv run python scripts/smoke_pipeline.py --device cpu --output artifacts/smoke
```

目录职责和安装后的模块入口见
[`docs/architecture.md`](docs/architecture.md)。

有可用 CUDA 时可将 `cpu` 改为 `cuda`。约 64M 参数的默认结构是 20 层、hidden size 512、
8 个 attention heads、2 个 KV heads、词表 6400；正式 CLI 默认上下文 768，并按命令中的
`--sequence-length` 构造模型。4GB 显存应从 batch size 1
开始；发生 OOM 时先把 sequence length 和 rollout group size 调低。

## 用真实数据运行完整链路

从仓库根运行时，训练命令默认读取 `dataset/` 根目录中的 JSONL。下列本地文件由 Git 和打包规则排除。

```text
dataset/pretrain_t2t_mini.jsonl   预训练 text
dataset/sft_t2t_mini.jsonl        多轮 conversations
dataset/dpo.jsonl                 chosen / rejected
dataset/agent_rl_math.jsonl       带 gt 的 GRPO 和 Agent 数学任务
```

已有全量语料可以用 `--data` 指定原路径，例如
`--data data/raw/minimind/pretrain/pretrain_t2t_mini.jsonl`。各阶段的字段和切分见
[`dataset/dataset.md`](dataset/dataset.md)。

默认 Hugging Face tokenizer 位于 `model/`，包含 `tokenizer.json` 和 `tokenizer_config.json`。
可用 `uv run python trainer/train_tokenizer.py` 训练独立的 SentencePiece 模型。后续训练和推理
需要沿用生成 checkpoint 时使用的 tokenizer。

输入任意文本查看 token ID、子词、压缩率和解码结果：

```bash
uv run python scripts/inspect_tokenizer.py \
  --text "你好，中国的首都是北京。MiniScale正在学习Agent RL！" \
  --add-bos \
  --add-eos
```

`vocab_size` 应为 `6400`，`round_trip` 应为 `true`。`characters_per_token` 越高表示这段文本压缩得越好，但应在固定的
中英/代码/数学测试集上比较，而不是只看一个句子。ByteLevel BPE 的 `tokens` 可能显示成
`ä½łå¥½` 一类内部字节符号，这是正常表示，判断正确性应看 `decoded` 和 `round_trip`。

然后逐阶段运行。下面的步数用于首次端到端实验，不代表最终收敛配置；每个阶段必须使用上个
阶段的权重和同一个 `model/` 目录：

正式预训练前建议先完整扫描语料。报告会给出内容指纹、有效/异常行、exact duplicate、真实 token
总量、稳定 train/validation 切分和 packing 利用率，但不会静默删除数据：

```bash
uv run python scripts/audit_data.py pretrain \
  --data dataset/pretrain_t2t_mini.jsonl \
  --sequence-length 768 \
  --output artifacts/data-audit.json
```

```bash
uv run python trainer/train_pretrain.py --steps 10000 --batch-size 1 \
  --num-hidden-layers 20 \
  --gradient-accumulation 16 --sequence-length 768 \
  --precision bf16 \
  --learning-rate 3e-4 --min-learning-rate 3e-5 \
  --warmup-steps 200 --save-every 500 --keep-last 3 \
  --validation-every 200 --validation-batches 100 \
  --shuffle-buffer-size 8192 \
  --output artifacts/pretrain

uv run python trainer/train_full_sft.py --steps 3000 --batch-size 1 \
  --gradient-accumulation 16 --max-length 512 \
  --precision bf16 \
  --learning-rate 2e-5 --min-learning-rate 2e-6 \
  --warmup-steps 100 \
  --validation-every 200 --validation-batches 100 \
  --save-every 500 --keep-last 3 --generation-every 1000 \
  --checkpoint artifacts/pretrain/best.pt \
  --output artifacts/sft

uv run python trainer/train_dpo.py --steps 1000 --batch-size 1 \
  --gradient-accumulation 16 --max-length 512 \
  --precision bf16 \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --beta 0.1 \
  --warmup-steps 50 \
  --validation-every 100 --validation-batches 100 \
  --save-every 200 --keep-last 3 \
  --checkpoint artifacts/sft/sft.pt \
  --output artifacts/dpo

uv run python scripts/audit_data.py grpo \
  --data dataset/agent_rl_math.jsonl \
  --output artifacts/grpo-data-audit.json

uv run python trainer/train_grpo.py --steps 500 --batch-size 1 --group-size 4 \
  --policy-epochs 2 --max-new-tokens 96 \
  --precision bf16 --reference-device cpu \
  --validation-every 100 --validation-prompts 100 \
  --save-every 100 --keep-last 3 \
  --checkpoint artifacts/dpo/dpo.pt \
  --output artifacts/grpo

uv run python scripts/audit_data.py agent \
  --data dataset/agent_rl_math.jsonl \
  --output artifacts/agent-data-audit.json

uv run python trainer/train_agent.py --steps 500 --batch-size 1 --group-size 2 \
  --policy-epochs 2 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --reference-device cpu \
  --validation-every 50 --validation-prompts 50 \
  --save-every 50 --keep-last 3 \
  --checkpoint artifacts/grpo/best.pt \
  --output artifacts/agent-rl
```

正式 SFT 会把 conversation 展开为 assistant-turn 样本，只监督当前 assistant 回复；训练数据使用
全局确定性 permutation，而不是只读取 JSONL 前部。默认按 conversation hash 留出 0.5% validation、
跳过 exact duplicate，并在超过 context 时同时保留最近 prompt 和回复开头。训练前建议先运行
`uv run python scripts/audit_data.py sft`。reasoning mask、旧 10k 权重启动、3050 Ti/5070 配置和精确恢复命令见
[`docs/sft.md`](docs/sft.md)。

正式 DPO 会校验 chosen/rejected 的共享 prompt，执行 pair-aware 安全截断和 prompt-hash 验证切分，
并支持 BF16、梯度累计、best/periodic/final checkpoint、固定 reference 与精确恢复。MiniMind 数据
审计、3050 Ti/5070 配置、指标解释和恢复命令见 [`docs/dpo.md`](docs/dpo.md)。

正式 GRPO/Agent-RL 同样支持稳定验证切分、BF16、冻结 reference、重复 policy epoch、best/periodic/
final checkpoint、W&B 和严格恢复。数学 verifier、防 reward hacking、3050 Ti/5070 配置见
[`docs/grpo.md`](docs/grpo.md)；工具能力审计、多轮环境和真实计算器推理见
[`docs/agent_rl.md`](docs/agent_rl.md)。跨阶段固定验证集比较与发布选择见
[`docs/evaluation.md`](docs/evaluation.md)。

预训练前 200 step 线性 warmup 到 `3e-4`，之后 cosine decay 到 `3e-5`。每 200 step
计算 `validation_loss` 和 `perplexity`；只要 validation loss 创历史最低，就立即覆盖 `best.pt`。
每 500 step 在 `checkpoints/` 保存完整训练状态，并只保留最近 3 个周期 checkpoint；每 1000
step 还会对固定中文、英文和 Python prompt 做 greedy generation evaluation。训练结束总会保存
`final.pt`。目录结构如下：

```text
artifacts/pretrain/
├── checkpoints/
│   └── step_xxxxxxxx.pt
├── generations/
│   └── step_xxxxxxxx.json
├── pretrain_metrics.jsonl
├── pretrain_run.json
├── best.pt
└── final.pt
```

`best.pt`、`final.pt` 和周期 checkpoint 都包含模型、AdamW、scheduler、step、token 计数、
`best_val_loss` 及 Python/NumPy/PyTorch/CUDA RNG 状态，可以用于断点续训。`pretrain_run.json`
保存解析后的完整 recipe、模型配置和数据/tokenizer 内容指纹。`keep_last` 只清理周期
checkpoint，不会删除 `best.pt`、`final.pt` 或 generations。安全边界收到 `Ctrl+C` 时会写入
`emergency_step_XXXXXXXX.pt`。如果中断时本步正在更新或验证，程序会指向最近成功写入的 checkpoint。
完整 checkpoint 通常约 700–800MB，请预留磁盘空间。

从周期或 emergency checkpoint 继续时，`--steps` 仍表示原计划的总步数，其他影响训练轨迹的
参数必须与原命令一致：

```bash
uv run python trainer/train_pretrain.py --steps 10000 --batch-size 1 \
  --num-hidden-layers 20 \
  --gradient-accumulation 16 --sequence-length 768 \
  --precision bf16 \
  --learning-rate 3e-4 --min-learning-rate 3e-5 \
  --warmup-steps 200 --save-every 500 --keep-last 3 \
  --validation-every 200 --validation-batches 100 \
  --generation-every 1000 \
  --resume artifacts/pretrain/checkpoints/step_00000500.pt \
  --output artifacts/pretrain
```

恢复时数据流会跳过 checkpoint 已消费的 micro-batches，metrics 文件中晚于恢复 step 的旧记录
会被移除，避免重复 step。optimizer、scheduler、step、token 计数、历史最佳 validation loss 和
RNG 都从断点恢复，因此 warmup/cosine schedule 会从原位置继续。旧版仅包含模型权重的
`pretrain.pt` 没有 optimizer/scheduler 状态，不能用 `--resume`；但仍可用于生成或作为后续
SFT 的初始权重。

严格恢复会在加载训练状态前校验数据、tokenizer、模型和影响训练轨迹的参数。布局迁移前的 v2
完整 checkpoint 可以直接恢复。更早的预训练 checkpoint 缺少这些指纹，需要人工确认输入未变后增加
`--allow-legacy-resume`；之后保存的新 checkpoint 不再需要该参数。完整格式和迁移约定见
[`docs/checkpointing.md`](docs/checkpointing.md)。新训练也不能直接复用已有产物的输出目录；应使用
新目录或显式 `--resume`，避免悄悄拼接 metrics 或覆盖 checkpoint。

默认精度是 FP32。上例中的 BF16 必须由支持 BF16 的 CUDA GPU 显式启用；它使用 autocast，但模型
参数与 AdamW state 仍为 FP32，也不使用 GradScaler。不支持的硬件会在训练前明确失败。优化器只对
Linear matrix weights 使用 weight decay，RMSNorm 与 tied embedding/lm head 不衰减；残差输出投影按
层数缩放初始化。完整训练语义见 [`docs/pretraining.md`](docs/pretraining.md)。

### 流式数据顺序

预训练 JSONL 使用 `IterableDataset`，不能直接给 DataLoader 设置 `shuffle=True`。训练集默认用
8192 个 packed sequences 的确定性 shuffle buffer，避免按语种、来源或网页类型排列的数据在
某个 step 突然整体切换；同一 seed 下顺序可复现，断点恢复会重新定位到完全相同的数据位置。
`--shuffle-buffer-size 0` 可以恢复旧的顺序读取行为，但不建议用于正式训练。

内置 validation split 是稳定的 hash 切分，适合纵向比较；如果准备了覆盖中文、英文、代码等
类别的独立 held-out JSONL，建议通过 `--validation-data dataset/pretrain_validation.jsonl`
传入。无论使用内置 split 还是独立文件，程序都会完整扫描 validation stream，并用固定 seed 的
reservoir sampling 等概率抽取候选 block，而不是只读取文件前部。`--validation-batches 100` 表示
缓存并复用最多 100 个 validation batch；实际 block 数为 `100 × batch_size`，候选不足时使用全部。

### W&B 训练曲线

W&B 是可选依赖，先安装并登录：

```bash
uv sync --extra tracking
uv run wandb login
```

向 `lotus111/MiniScale` 记录训练 loss、validation loss、perplexity、learning rate、grad norm、
tokens seen 和 generation tables：

```bash
uv run python trainer/train_pretrain.py --steps 10000 --batch-size 4 \
  --num-hidden-layers 20 \
  --gradient-accumulation 4 --sequence-length 768 \
  --learning-rate 3e-4 --min-learning-rate 3e-5 \
  --warmup-steps 200 --validation-every 200 \
  --save-every 500 --keep-last 3 --generation-every 1000 \
  --shuffle-buffer-size 8192 \
  --wandb --wandb-project MiniScale --wandb-entity lotus111 \
  --wandb-run-name pretrain-64m-shuffled-bs4 \
  --wandb-retry-every 200 \
  --output artifacts/pretrain-shuffled
```

启用 W&B 时，run ID 会写入完整 checkpoint。续训命令使用 `--resume` 并带上 `--wandb`，
训练会从 checkpoint 读出 ID，接回同一个 W&B run。首次恢复缺少 run ID 的旧 checkpoint 时，
可以显式传入 `--wandb-run-id`。

待上传指标先写到输出目录的 `wandb_pending.jsonl`。重新启动时沿用同一个 `--output`，即可继续
处理这个队列。断联后默认每 200 step 尝试重连，可用 `--wandb-retry-every` 调整间隔。标量和
generation table 分开上传，标量优先补传。也可以使用 `--wandb-mode offline`，联网后运行
`uv run wandb sync <离线 run 目录>`。

当前 GRPO 默认使用 `agent_rl_math.jsonl`，因为其中的 `gt` 能形成确定的 verifier reward。全量审计
发现 20,000 行中有 19,430 个去重有效任务，默认稳定切分为 18,484 train / 946 validation。
`rlaif.jsonl` 是开放式回答，不能拿“有没有数字”充当质量奖励；要使用它，应另接冻结的
reward model 或 LLM-as-a-judge。混合 `agent_rl.jsonl` 还包含当前 calculator sandbox 不支持的
工具，所以 Agent RL 也默认使用数学子集。数学数据中附带的天气、翻译、汇率等非计算器 schema 会在
加载边界被剥离并写入审计报告，不会暴露给只能执行计算器的环境。

训练终端使用统一的紧凑格式，只显示 loss、LR、gradient norm、吞吐、CUDA peak memory 和少量
阶段关键指标；gradient clipping 发生时会在 grad 后显示 `*`。完整累计计数、性能和验证统计仍原样
追加写入 `artifacts/<stage>/*_metrics.jsonl`，供后续分析和画图。另开终端可以这样观察：

```text
[pretrain] step 1900 | loss 3.218 | lr 3.00e-4 | grad 0.79 | tok/s 39.9k | mem 5290MB
```

```bash
tail -f artifacts/pretrain/pretrain_metrics.jsonl
```

预训练主要看 `train_loss` 长期下降、`validation_loss` 同步下降，以及 `perplexity` 没有持续
反弹；单个 step 的抖动正常。DPO 看 loss 和 preference accuracy，GRPO/Agent RL 看 reward、
KL、clip fraction、success/tool-call rate。只有写日志的更新步会测量吞吐和 CUDA 峰值显存。
预训练和 DPO 默认关闭周期生成，可用 `--generation-every N` 启用。SFT 仍按固定生成套件选模。

验证 CPU 流水线时使用 smoke 入口：

```bash
uv run python scripts/smoke_pipeline.py --device cpu --output artifacts/smoke
```

输出在 `artifacts/`，该目录已被 Git 忽略。Smoke 用微型 JSONL 调用正式预训练和 SFT 函数，流水线清单
记录各阶段 loss、reward、耗时和 checkpoint 路径。

使用真实 SFT 或最后一个 Agent RL checkpoint 生成文本时必须传入训练时的 tokenizer：

```bash
uv run python eval_llm.py \
  --checkpoint artifacts/agent-rl/best.pt \
  --tokenizer model \
  --prompt "请计算 7109*2920，只给最终结果。" \
  --calculator --max-turns 3 \
  --temperature 0 --max-new-tokens 64

# --raw 只打印模型回答
uv run python eval_llm.py \
  --checkpoint artifacts/smoke/agent_rl.pt \
  --prompt "Use the calculator for 3*4." \
  --calculator \
  --temperature 0 \
  --raw
```

`best.pt`、`final.pt`、`sft.pt`、`dpo.pt`、`rl.pt` 和 `agent_rl.pt` 都包含可加载的模型权重；
评估 base 模型 `best.pt` 或 `final.pt` 的续写时可给 `eval_llm.py` 增加 `--raw-prompt`。
但通常优先选择 `sft.pt` 或 `agent_rl.pt`。训练步数和数据量决定输出质量，文件能加载不等于
模型已经具备泛化能力。

## 建议学习顺序

1. 从 `model/model_miniscale.py` 和 `model/tokenizer.py` 推导张量 shape，确认 causal test 为什么成立。
2. 在 `dataset/lm_dataset.py` 和 `trainer/train_pretrain.py` 跟踪流式读取、packing、shift label 和 next-token loss。
3. 在 `trainer/train_full_sft.py` 跟踪 assistant-only 监督和 `-100` mask。
4. 在 `trainer/train_dpo.py` 检查 policy 与 reference 对 chosen/rejected 的相对 log-probability。
5. 从 `trainer/train_grpo.py` 跟到 `trainer/rl/objective.py`，依次看组内优势、ratio clipping 和 reference KL。
6. 从 `trainer/train_agent.py` 跟到 `trainer/rl/rollout.py` 和 `trainer/rl/environment.py`，观察工具响应如何进入上下文以及 action mask 如何标记模型动作。
7. 修改一个组件并补测试，例如新增 `search` mock 工具、格式奖励或离线 eval；这才会逐渐变成
   你的工程，而不是 MiniMind 的复刻。

## 与工业训练栈的边界

这个小型实现保留了阶段化完整 checkpoint、严格恢复、assistant/action masking、old/reference
policy、稳定 held-out validation、可验证 reward、受限工具执行和端到端测试。但真正扩大训练规模前
还需要：

- 为现有 BPE/JSONL 数据加入去重、质量过滤、数据配比和评测污染检测；
- 加入 gradient checkpointing、显式 FlashAttention backend、FSDP/DeepSpeed 和 MFU 统计；
- 将 rollout 与 learner 解耦，用 vLLM/SGLang 一类推理服务异步采样，并处理 policy weight 同步；
- 对工具执行使用进程/容器级 sandbox、超时、资源限额和审计，而不仅是当前的 AST 白名单；
- 建立覆盖通用聊天回归的跨阶段 benchmark、pass@k 和最终发布门禁；
- 将规则 verifier 扩展到更多可验证领域，并持续扩充 reward-hacking 失败样本。

求职展示时，应把 `pipeline_manifest.json`、测试、消融实验和失败案例作为证据；“能运行”只是
第一层，能解释 reward、mask、KL、采样吞吐和评测可信度才是 Agentic RL 岗位更看重的部分。

## 目录

```text
model/                  模型配置、decoder、tokenizer 和词表资源
dataset/                JSONL 读取、切分、批处理、评估套件和质量策略
trainer/                五个训练阶段、tokenizer 训练和共享运行机制
scripts/                数据工具、评测、推理实现、环境检查和 smoke
eval_llm.py             根目录推理入口
tests/                  算法语义与端到端回归测试
artifacts/              本地 checkpoint/manifest（不提交）
```

贡献流程、测试约定和 checkpoint 兼容性要求见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。CI 会在 push 和
pull request 上使用锁定依赖运行完整 unittest。公开发布前仍需由仓库所有者明确选择并添加许可证。

## 旧命令迁移

从源码目录可直接运行右列脚本，也可以使用对应模块，例如
`uv run python -m trainer.train_full_sft --help`。安装 wheel 后，推理入口是
`python -m scripts.inference`。

| 旧入口 | 新命令 |
| --- | --- |
| `uv run miniscale pretrain` | `uv run python trainer/train_pretrain.py` |
| `uv run miniscale sft` | `uv run python trainer/train_full_sft.py` |
| `uv run miniscale dpo` | `uv run python trainer/train_dpo.py` |
| `uv run miniscale grpo` | `uv run python trainer/train_grpo.py` |
| `uv run miniscale agent-rl` | `uv run python trainer/train_agent.py` |
| `uv run miniscale train-tokenizer` | `uv run python trainer/train_tokenizer.py` |
| `uv run miniscale tokenize` | `uv run python scripts/inspect_tokenizer.py` |
| `uv run miniscale generate`、`uv run python generate.py` | `uv run python eval_llm.py` |
| `uv run miniscale evaluate` | `uv run python scripts/evaluate_checkpoints.py rl` |
| `uv run miniscale evaluate-sft` | `uv run python scripts/evaluate_checkpoints.py sft` |
| `uv run miniscale audit-pretrain-data` | `uv run python scripts/audit_data.py pretrain` |
| `uv run miniscale audit-sft-data` | `uv run python scripts/audit_data.py sft` |
| `uv run miniscale audit-dpo-data` | `uv run python scripts/audit_data.py dpo` |
| `uv run miniscale audit-grpo-data` | `uv run python scripts/audit_data.py grpo` |
| `uv run miniscale audit-agent-data` | `uv run python scripts/audit_data.py agent` |
| `uv run miniscale prepare-sft-data` | `uv run python scripts/prepare_data.py` |
| `uv run miniscale doctor` | `uv run python scripts/doctor.py` |
| `uv run miniscale pipeline` | `uv run python scripts/smoke_pipeline.py` |
| `uv run python trainer/train_sft.py` 及其他旧阶段 smoke 脚本 | `uv run python scripts/smoke_pipeline.py` |

## 阶段提交

本仓库按可验证里程碑提交：预训练、SFT、GRPO、Agent RL、最终 CLI/文档。可用
`git log --oneline` 审核历史，用 `git show <commit>` 查看某阶段完整改动。
