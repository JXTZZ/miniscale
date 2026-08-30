# Supervised fine-tuning

MiniScale 的正式 SFT 路径把一个 conversation 展开为多个 assistant-turn 样本。每个样本保留截至
当前 assistant 回复的对话上下文，但只监督当前回复；user、system、tool observation、历史 assistant
回复和 role header 都使用 `-100` mask。模型仍在内部完成 next-token shift。

## 训练前审计

先完整扫描结构、normalized exact duplicate、稳定切分、身份字符串、回复重复率和类别分布，再对全局
随机样本计算 token 长度与截断：

```bash
uv run miniscale audit-sft-data \
  --data data/derived/sft/sft_miniscale_lotusy_v2.jsonl \
  --tokenizer data/tokenizer/minimind \
  --max-length 768 \
  --target-mode response_only \
  --validation-fraction 0.005 \
  --sample-size 5000 \
  --identity-pattern MiniMind \
  --identity-pattern MiniScale \
  --identity-pattern jingyaogong \
  --identity-pattern LoTusY \
  --output artifacts/sft-miniscale-lotusy-v2-audit-768-response-only.json
```

审计只生成报告，不改写原始数据，也不会根据身份字符串自动过滤。正式训练默认按 canonical
conversation identity 跳过 exact duplicate，并把去重数量写入 `sft_run.json`。如果需要过滤或重写
身份回答，应使用独立数据准备命令生成新的 JSONL，而不是在训练循环里静默修改。下面的替换对
conversation 的所有字符串值大小写不敏感，并在替换后重新执行 exact dedup：

```bash
uv run miniscale prepare-sft-data \
  --data data/derived/sft/sft_miniscale_lotusy.jsonl \
  --tokenizer data/tokenizer/minimind \
  --replace-pattern MiniMind=MiniScale \
  --replace-pattern jingyaogong=LoTusY \
  --quality-policy data/policies/sft_quality_v1.json \
  --output data/derived/sft/sft_miniscale_lotusy_v2.jsonl
```

质量策略会做 Unicode/空白归一化 exact dedup、按 prompt/response/短回复/身份回复限频、过滤严重循环、
限制过长目标、执行类别配额，再用固定 seed 的 hash reservoir 选出最多 25 万个监督目标。命令拒绝覆盖
源文件或已有输出，并同时写入 `.manifest.json`，记录输入/输出 SHA-256、策略和每类接受/拒绝计数。
输出行中的 `sft_selection.target_positions` 明确指定参与训练的 assistant turn，原始文件保持不变。

## 数据顺序、切分和截断

启动时会扫描 JSONL，建立 assistant-turn 的 byte-offset 索引。DataLoader 对完整索引执行固定 seed
的全局 permutation，因此短训练不会只读取文件前部。相同 seed、数据和 worker 数得到相同顺序。

默认用 canonical conversation hash 将 0.5% conversation 放入 validation；exact duplicate 总是属于
同一个 split。也可以通过 `--validation-data` 使用独立 JSONL。程序固定随机抽取 validation 样本并在
每次验证时复用；`--validation-batches 100 --batch-size 1` 就是固定 100 个随机 held-out 样本。

样本超过 context 时，程序保留最近的 prompt context 和 assistant 回复开头，不再无条件保留最后
512 token。默认至少保留 32 个上下文 token；如果回复本身过长，会截断回复尾部。训练样本绝不会从
supervised token 开始。

## Reasoning 监督策略

- `--target-mode reasoning_and_response`：监督 `<think>`、reasoning、最终回复和 assistant 结束符；
- `--target-mode response_only`：目标 assistant 的 reasoning 会从输入中移除，只监督最终回复和结束符。

默认仍是 `reasoning_and_response`。针对当前 64M 模型的通用对话阶段，推荐显式使用
`--target-mode response_only`：小模型优先学习终止、简洁回答和格式遵循，降低不稳定思维文本带来的循环。
如要训练显式推理能力，应另做高质量、可验证的 reasoning 子集实验，不要直接混用全部原始
`reasoning_content`。该选择属于严格 resume identity，不能在同一个 run 中途切换；后续 DPO 必须使用
相同的 `--target-mode`。

## 从四轮预训练权重开始

预训练 `best.pt` 可以作为 SFT 的初始化权重。它不是 SFT optimizer checkpoint，所以应使用
`--checkpoint`，不能使用 `--resume`。`--max-length` 不能超过预训练 checkpoint 的 context length。

先进行 20 step 验证：

```bash
uv run miniscale sft \
  --steps 20 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --max-length 768 \
  --precision bf16 \
  --learning-rate 2e-5 \
  --min-learning-rate 2e-6 \
  --warmup-steps 2 \
  --validation-every 10 \
  --validation-batches 10 \
  --save-every 10 \
  --keep-last 2 \
  --num-workers 0 \
  --device cuda \
  --target-mode response_only \
  --generation-every 10 \
  --generation-suite data/eval/sft_generation_v1.jsonl \
  --checkpoint artifacts/pretrain-20l-768-4epoch/best.pt \
  --output artifacts/sft-v2-smoke
```

正式审计得到 230,666 个 train target；effective batch 16 时，一轮约 14,417 optimizer steps。
下面给出 18,000 step（约 1.25 轮）上限，并在 12,000 step 后允许质量早停；上限不是要求必须跑满：

```bash
uv run miniscale sft \
  --data data/derived/sft/sft_miniscale_lotusy_v2.jsonl \
  --checkpoint artifacts/pretrain-20l-768-4epoch/best.pt \
  --output artifacts/sft-64m-response-v2 \
  --steps 18000 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --target-mode response_only \
  --precision bf16 \
  --learning-rate 1e-5 \
  --warmup-steps 500 \
  --early-stopping-patience 4 \
  --early-stopping-min-steps 12000 \
  --wandb \
  --wandb-run-name sft-64m-response-v2
```

省略项使用经过验证的默认值：checkpoint context 768、min LR `2e-6`、每 200 step 验证、100 个
validation batch、每 500 step 保存、每 1000 step 运行固定生成质量集、保留最近 3 个周期 checkpoint、
W&B project `MiniScale` 和自动选择 CUDA。早停默认关闭，所以上面的 patience/min-steps 两项不能省略。

若 `8 × 2` 显存不足，退到 `4 × 4` 或 `2 × 8`，保持每个 optimizer update 仍为 16 个样本。
不要因为更换 micro-batch 而修改 LR、数据 seed 或总 steps。BF16 会在启动时检查硬件支持，模型参数与
AdamW state 保持 FP32。

## Checkpoint 与恢复

输出目录包含：

```text
artifacts/sft-64m-response-v2/
├── checkpoints/step_XXXXXXXX.pt
├── generations/step_XXXXXXXX.json
├── sft_metrics.jsonl
├── sft_run.json
├── best_loss.pt
├── best_quality.pt
├── best.pt -> best_quality.pt 的兼容镜像
└── sft.pt
```

`best_loss.pt` 按固定 validation loss 选择，`best_quality.pt` 按固定 raw-greedy 生成质量选择，`best.pt`
兼容指向后者。周期 checkpoint 和最终 `sft.pt` 都包含模型、AdamW、scheduler、step、数据进度、
Python/NumPy/PyTorch/CUDA RNG、输入/监督 token 计数和严格 resume identity。`sft.pt` 仍保留顶层
`config` 与 `model`，所以 generate、DPO 和旧的 `load_checkpoint` 调用方式不变。

精确恢复时，`--steps` 表示原计划总步数，所有影响训练轨迹的参数必须和首次命令一致：

```bash
uv run miniscale sft \
  --data data/derived/sft/sft_miniscale_lotusy_v2.jsonl \
  --steps 18000 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --max-length 768 \
  --target-mode response_only \
  --precision bf16 \
  --learning-rate 1e-5 \
  --min-learning-rate 1e-6 \
  --warmup-steps 500 \
  --validation-every 500 \
  --validation-batches 100 \
  --save-every 1000 \
  --keep-last 3 \
  --generation-every 500 \
  --generation-suite data/eval/sft_generation_v1.jsonl \
  --early-stopping-patience 4 \
  --early-stopping-min-steps 12000 \
  --num-workers 0 \
  --device cuda \
  --wandb \
  --wandb-project MiniScale \
  --resume artifacts/sft-64m-response-v2/checkpoints/step_00001000.pt \
  --output artifacts/sft-64m-response-v2
```

如果想改变总 steps、LR 或数据策略，应把已有 SFT checkpoint 当作新实验的 `--checkpoint`；这会重置
optimizer/scheduler，并在新 manifest 中记录父权重内容指纹。新训练拒绝覆盖已有输出，精确 resume
会先校验模型、数据、tokenizer、mask、截断、采样、optimizer、schedule、batch、seed 和精度。

## 指标

训练日志和 W&B 会记录 train/validation loss、perplexity、validation token accuracy、LR、gradient
norm、gradient clipping 比例、input/supervised tokens、examples seen、吞吐和 CUDA peak memory。
固定 raw-greedy generation 还记录任务通过率、EOS/max-length/循环率、重复 n-gram、prompt echo、特殊
token 和 think 泄漏，并参与 `best_quality.pt` 与可选 early stopping 的选择。

训练结束后用同一套 24 条固定探针同时比较原始 greedy 和部署防循环解码：

```bash
uv run miniscale evaluate-sft \
  --checkpoint artifacts/sft-64m-response-v2/best_loss.pt \
  --checkpoint artifacts/sft-64m-response-v2/best_quality.pt \
  --suite data/eval/sft_generation_v1.jsonl \
  --tokenizer data/tokenizer/minimind \
  --precision bf16 \
  --device cuda \
  --output artifacts/sft-64m-response-v2/checkpoint-comparison.json
```

部署生成推荐从 `--temperature 0.6 --top-k 50 --top-p 0.9 --repetition-penalty 1.1
--no-repeat-ngram-size 4` 开始；这些参数只能抑制解码循环，不能替代数据清洗和 checkpoint 选择。
