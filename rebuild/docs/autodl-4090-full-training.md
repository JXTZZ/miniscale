# AutoDL 单卡 RTX 4090：从克隆分支到全链路训练

本文针对 `learn/rebuild-from-scratch` 分支的 `rebuild/`，依次运行 **Pretrain → SFT → DPO → 数学 GRPO → 计算器 Agent RL**，最后做固定验证集对比、生成和 Hugging Face 导出。Pretrain 和 SFT 使用 ModelScope 的全量 `pretrain_t2t.jsonl`、`sft_t2t.jsonl`；DPO、GRPO 和 Agent RL 沿用各自数据文件。默认模型约 63.6M 参数、上下文 768；使用仓库自带的 MiniMind 6400 词 tokenizer。这里的“尽量一轮”指预训练读完约一轮 packed block、SFT 读完约一轮去重后的训练样本；DPO 和在线 RL 使用单独的实验预算。**全量数据可能需要很长租机时间，实际耗时须依据冒烟测试后的实测吞吐估算；本文不保证最终质量或特定吞吐。**

只使用 `agent_rl_math.jsonl` 训练后两阶段。下载脚本也会取回 `rlaif.jsonl` 和混合 `agent_rl.jsonl`，但当前没有开放式 RLAIF 的可靠奖励模型，也没有混合 Agent 文件所需的全部真实工具环境，不要把这两份文件直接喂给 GRPO/Agent RL。

## 1. 创建实例并确认资源

以下命令假设使用 AutoDL **普通容器实例**、一张 RTX 4090，且数据盘挂载在 `/root/autodl-tmp`。选择能运行本项目锁定的 PyTorch 2.9.1 + CUDA 12.8 wheel 的主机驱动；NVIDIA 列出的 CUDA 12.8 GA Linux 驱动版本为 570.26 起，最终仍以安装后的 CUDA 自检为准。4090 有 24 GB 显存；下面从保守的 micro-batch 开始，正式长跑前要做同配置的五阶段冒烟测试。[4090 规格](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、[CUDA 12.8 驱动说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)。

建议给数据盘至少 **120 GB 可用空间**：全量 pretrain 约 8.28 GB、全量 SFT 约 14.10 GB，六份训练链数据合计约 22.55 GB；此外还要放 `uv` 缓存、虚拟环境、多阶段 checkpoint 和日志。`df -h` 应显示实际剩余空间，下载前后都检查。全量审计和 SFT 索引会占用较多**系统内存**，建议选 64 GB RAM 或更高的实例，并用 `free -h` 监控；24 GB 显存只决定每步可用的 micro-batch，文件大小主要增加训练时长、磁盘和 CPU/RAM 负担。代码、数据和训练产物都放在数据盘。AutoDL 说明普通容器重置系统或更换镜像时会清空系统盘，但保留 `/root/autodl-tmp`；本地数据盘没有冗余保障，重要 checkpoint 仍要备份。[实例数据说明](https://www.autodl.com/docs/instance_data/)、[本地数据盘说明](https://www.autodl.com/docs/local_disk/)。

打开实例终端或 SSH，先检查：

```bash
nvidia-smi
df -h /root/autodl-tmp
free -h
```

SSH 断开可能终止前台训练，所以长任务在 `tmux` 中运行；AutoDL 也建议使用 `screen`/`tmux`。[守护进程说明](https://www.autodl.com/docs/daemon/)。若镜像没有 `tmux`，在实例中安装：

```bash
apt-get update && apt-get install -y tmux git curl
```

## 2. 克隆指定分支并建立独立环境

以下所有项目命令都在 **`rebuild/` 目录**执行，避免导入仓库根目录的旧版同名包。

```bash
cd /root/autodl-tmp
git clone --branch learn/rebuild-from-scratch --single-branch https://github.com/JXTZZ/miniscale.git
cd miniscale/rebuild
mkdir -p artifacts/autodl-4090/logs
git branch --show-current
git rev-parse HEAD | tee artifacts/autodl-4090/git-commit.txt
```

如果镜像已有 `uv`，跳过安装行。下面的 `uv sync --frozen --extra tracking` 使用仓库的 `uv.lock` 和 CUDA 12.8 PyTorch wheel；不要混用镜像自带的系统 Python/Torch。`uv` 安装命令来自[官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="/root/.local/bin:$PATH"
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
uv python install 3.12
uv sync --frozen --extra tracking
uv run python -c 'import miniscale; print(miniscale.__file__)'
uv run miniscale doctor
uv run python - <<'PY'
import torch
print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PY
uv run python -m unittest discover -s tests -q
```

导入路径应包含 `/rebuild/src/miniscale/`，GPU 应显示 4090，测试应全部通过。如果 CUDA 自检失败，先核对 AutoDL 主机驱动、GPU 分配和锁定 wheel；此时不要启动训练或擅自改动 `uv.lock`。换新终端后重新 `cd` 到 `rebuild/` 并设置 `UV_CACHE_DIR`。

### W&B 曲线：正式训练前登录一次

`tracking` 是项目的可选依赖；上面的 `uv sync --frozen --extra tracking` 已把它安装到项目环境。正式训练命令启用 `--wandb`，先在 AutoDL 终端登录你自己的 W&B 账号：

```bash
uv run wandb login
```

按终端提示完成认证，不要把 API key 写进脚本、仓库或聊天。五个阶段各建一个 run，默认写到登录账号下的 `MiniScale` 项目；如需团队空间，在正式命令加 `--wandb-entity 你的团队名`。打开 W&B 项目页面，选择对应 run，以训练 `step` 为横轴查看 `train/loss`、`train/learning_rate`；预训练、SFT、DPO 还可看 `eval/loss`，GRPO 看 `eval/reward` 与 `eval/exact_match`，Agent RL 看 `eval/success_rate`。验证指标只在 `--validation-every` 指定的步骤出现，学习率和训练 loss 每个训练 step 都会写入。[W&B 指标说明](https://docs.wandb.ai/guides/track/log/)。

如果不使用 W&B，删掉正式训练命令中的 `--wandb` 那一行即可；`*_metrics.jsonl` 仍会完整保存。第 7 节有无需 W&B 账号的本地 PNG 绘图命令。W&B 网络上传失败时训练会继续，未上传的事件会先留在输出目录的 `wandb_pending.jsonl`，默认每 200 step 重试；请同时保留这个文件和 `wandb/` 目录。启用 W&B 会上传指标及固定生成探针表；不想上传时可使用本地绘图。

## 3. 下载、审计并计算一轮预算

全量下载和审计可能持续很久，先单独执行 `tmux new -s miniscale`。进入新会话后运行下面的命令；按 `Ctrl+b`、`d` 可暂时离开，重新登录后用 `tmux attach -t miniscale` 返回。

```bash
cd /root/autodl-tmp/miniscale/rebuild
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
set -o pipefail
```

下载脚本的 `--profile full` 会获取两份全量文件及 DPO/RL/Agent 的四份文件，共约 **22.55 GB**（ModelScope 2026-09-29 的文件元数据；上游更新后以脚本打印的总量为准）。默认不带参数仍下载 mini，**本配方必须加 `--profile full`**。脚本支持 `.part` 续传，并按上游 SHA-256 验证；请保留 `download_manifest.json`。运行中不要替换数据或 tokenizer，否则严格 resume 会拒绝。[ModelScope 数据文件列表](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。

```bash
uv run python scripts/download_minimind.py --profile full
du -sh data/raw/minimind
test -s data/raw/minimind/download_manifest.json
test -s data/raw/minimind/pretrain/pretrain_t2t.jsonl
test -s data/raw/minimind/sft/sft_t2t.jsonl
uv run python -c "from miniscale.tokenizer import load_tokenizer; t=load_tokenizer('data/tokenizer/minimind'); print(type(t).__name__, t.vocab_size)"
```

最后一条应显示 `HuggingFaceTokenizer 6400`。随后扫描真实数据。SFT 选择 `response_only`；DPO 必须保持相同目标语义。全量预训练审计会分词并统计所有文本的 hash，全量 SFT 审计会建立索引并再次扫描所有行；它们可能耗时数小时并使用较多 RAM，不会修改原始 JSONL。执行时用 `free -h` 观察内存；审计完成后再计算预算。

```bash
uv run miniscale audit-pretrain-data \
  --data data/raw/minimind/pretrain/pretrain_t2t.jsonl --sequence-length 768 \
  --output artifacts/autodl-4090/pretrain-audit.json
uv run miniscale audit-sft-data \
  --data data/raw/minimind/sft/sft_t2t.jsonl \
  --max-length 768 --target-mode response_only \
  --sample-size 2000 --output artifacts/autodl-4090/sft-audit.json
uv run miniscale audit-dpo-data --max-length 768 --target-mode response_only \
  --sample-size 2000 --output artifacts/autodl-4090/dpo-audit.json
uv run miniscale audit-grpo-data --output artifacts/autodl-4090/grpo-audit.json
uv run miniscale audit-agent-data --output artifacts/autodl-4090/agent-audit.json
```

预训练审计报告包含训练 block 数。SFT 审计为了统计原始重复行，**不会去重**；正式 SFT 默认会去重，所以用训练器的同一个索引计算训练样本数。下面把两阶段有效 batch 固定为 `4 × 4 = 16`，把预算保存为可重新 `source` 的文件：

```bash
uv run python - <<'PY'
import json
from math import ceil
from pathlib import Path
from miniscale.data.sft import SFTCorpusIndex

report = json.loads(Path('artifacts/autodl-4090/pretrain-audit.json').read_text())
blocks = report['tokens']['train']['packed_blocks']
index = SFTCorpusIndex.build(
    'data/raw/minimind/sft/sft_t2t.jsonl',
    validation_fraction=0.005, target_mode='response_only',
    destination='split', deduplicate_exact=True,
)
examples = index.stats.train_examples
pretrain_steps, sft_steps = ceil(blocks / 16), ceil(examples / 16)
Path('artifacts/autodl-4090/budget.env').write_text(
    f'PRETRAIN_STEPS={pretrain_steps}\nSFT_STEPS={sft_steps}\n'
)
print('pretrain blocks:', blocks, 'steps:', pretrain_steps)
print('SFT train examples:', examples, 'steps:', sft_steps)
PY
source artifacts/autodl-4090/budget.env
cat artifacts/autodl-4090/budget.env
printf 'Pretrain: %s optimizer updates; SFT: %s optimizer updates\n' "$PRETRAIN_STEPS" "$SFT_STEPS"
test "$PRETRAIN_STEPS" -gt 0 && test "$SFT_STEPS" -gt 0
```

`budget.env` 会写入形如 `PRETRAIN_STEPS=...` 和 `SFT_STEPS=...` 的**实际整数**。在当前 shell 执行 `source` 后，`--steps "$PRETRAIN_STEPS"` 才会被 shell 展开为该整数；另开 shell 时需要重新 `source`，并可用上面的 `cat` / `printf` 查看。`--steps` 指优化器更新次数，不是文件行数：预训练每步处理 `batch-size 4 × gradient-accumulation 4 = 16` 个 768-token packed block，因此 `PRETRAIN_STEPS = ceil(训练分区 packed_blocks / 16)`；SFT 每步处理 16 个去重后的训练目标，因此 `SFT_STEPS = ceil(训练目标数 / 16)`。末步不足一批时可能跨过一轮边界少量数据。

以实际审计和去重索引计算的 `PRETRAIN_STEPS`、`SFT_STEPS` 为准；不要套用 mini 数据的历史步数。0.5% 的原始文件数据固定留作验证，因此“约一轮”指训练分区约一轮，不是文件中每条都参与梯度更新，也不是收敛保证。全量 SFT 的步数可能远高于 README 的 3,000 步示例；先根据实测单步耗时计算租机预算，再决定是否启动完整长跑。

## 4. 五阶段真实数据冒烟测试

继续使用第 3 节的 `tmux` 会话；如果重新登录，请先 `tmux attach -t miniscale`。**以下 smoke 使用单独输出目录，其 checkpoint 只用于下一条 smoke；正式训练从头建立新目录。** 在会话中重新加载预算：

```bash
cd /root/autodl-tmp/miniscale/rebuild
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
source artifacts/autodl-4090/budget.env
set -o pipefail
```

下面分别触碰真实 JSONL、BF16、验证、DPO reference、RL rollout 和计算器环境。预训练/SFT/DPO 的生成探针在此关掉以节省时间；正式运行会启用。

```bash
uv run miniscale pretrain \
  --data data/raw/minimind/pretrain/pretrain_t2t.jsonl \
  --steps 2 --batch-size 4 --gradient-accumulation 4 \
  --sequence-length 768 --num-hidden-layers 20 --precision bf16 --device cuda \
  --warmup-steps 1 --validation-every 1 --validation-batches 2 \
  --generation-every 0 --save-every 0 \
  --output artifacts/autodl-4090/smoke/pretrain

uv run miniscale sft \
  --data data/raw/minimind/sft/sft_t2t.jsonl \
  --steps 2 --batch-size 4 --gradient-accumulation 4 \
  --max-length 768 --target-mode response_only --precision bf16 --device cuda \
  --warmup-steps 1 --validation-every 1 --validation-batches 2 \
  --generation-every 0 --save-every 0 \
  --checkpoint artifacts/autodl-4090/smoke/pretrain/final.pt \
  --output artifacts/autodl-4090/smoke/sft

uv run miniscale dpo --steps 2 --batch-size 2 --gradient-accumulation 8 \
  --max-length 768 --target-mode response_only --precision bf16 --device cuda \
  --warmup-steps 1 --validation-every 1 --validation-batches 2 \
  --generation-every 0 --save-every 0 \
  --checkpoint artifacts/autodl-4090/smoke/sft/sft.pt \
  --output artifacts/autodl-4090/smoke/dpo

uv run miniscale grpo --steps 2 --batch-size 1 --group-size 4 \
  --policy-epochs 2 --max-new-tokens 96 --precision bf16 --device cuda \
  --reference-device same --warmup-steps 1 \
  --validation-every 1 --validation-prompts 2 --save-every 0 \
  --checkpoint artifacts/autodl-4090/smoke/dpo/dpo.pt \
  --output artifacts/autodl-4090/smoke/grpo

uv run miniscale agent-rl --steps 2 --batch-size 1 --group-size 2 \
  --policy-epochs 2 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --device cuda --reference-device same --warmup-steps 1 \
  --validation-every 1 --validation-prompts 2 --save-every 0 \
  --checkpoint artifacts/autodl-4090/smoke/grpo/rl.pt \
  --output artifacts/autodl-4090/smoke/agent-rl
```

确认五条命令均以退出码 0 结束，目录中依次有 `final.pt`、`sft.pt`、`dpo.pt`、`rl.pt`、`agent_rl.pt`。若 OOM，先确认实例没有其他 GPU 进程；再在**正式训练启动前**降低对应阶段 micro-batch，保持预训练/SFT/DPO 的有效 batch 为 16，例如 `4×4 → 2×8 → 1×16`。RL 先将 `--reference-device same` 改成 `cpu`，或将 group size 从 4 降到 2。变更后重跑对应 smoke，正式命令也使用同一组参数。调整 micro-batch 会改变数据分批与严格 resume 身份；不要在运行中途直接改。

## 5. 正式训练：Pretrain → SFT → DPO

本配方直接用以下 CLI 参数作为训练配置：`--data` 指向全量文件，`--steps` 从 `budget.env` 读取；batch、梯度累计、学习率、验证频率等也都在命令中明确给出。项目代码里的 mini 默认路径留给原有用法；执行这些命令时，CLI 参数会覆盖默认值。先确认 `cat artifacts/autodl-4090/budget.env` 显示两个正整数。

在同一个 `tmux` shell 中运行。每条命令成功后再运行下一条；`set -o pipefail` 保证使用 `tee` 时训练失败仍返回非零状态。检查验证 loss、有限的梯度、显存和磁盘余量：

```bash
uv run miniscale pretrain \
  --data data/raw/minimind/pretrain/pretrain_t2t.jsonl \
  --tokenizer data/tokenizer/minimind \
  --output artifacts/autodl-4090/pretrain \
  --steps "$PRETRAIN_STEPS" --batch-size 4 --gradient-accumulation 4 \
  --num-hidden-layers 20 --sequence-length 768 --precision bf16 --device cuda \
  --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 200 \
  --validation-fraction 0.005 --validation-every 500 --validation-batches 50 \
  --generation-every 2000 --save-every 500 --keep-last 2 \
  --shuffle-buffer-size 8192 --num-workers 0 --seed 42 \
  --wandb --wandb-project MiniScale --wandb-run-name autodl4090-pretrain \
  2>&1 | tee artifacts/autodl-4090/logs/pretrain.log

test -s artifacts/autodl-4090/pretrain/final.pt
df -h /root/autodl-tmp
```

`final.pt` 是走完预算的权重；`best.pt` 是验证 loss 最低的权重。本配方用 `final.pt` 交给 SFT，便于追踪“一轮后”的模型，同时保留 `best.pt` 以便比较。预训练的验证数据是训练文件的固定 hash 切分，约 0.5% 不参与训练。长跑开始后可用 `tail -n 2 artifacts/autodl-4090/pretrain/pretrain_metrics.jsonl` 查看 `update_seconds`、`tokens_per_second` 和 `cuda_peak_memory_mb`；先测几十步再估算剩余时间，验证和生成还会产生额外耗时。然后训练全量 SFT 的一轮，不启用早停：

```bash
uv run miniscale sft \
  --data data/raw/minimind/sft/sft_t2t.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/autodl-4090/pretrain/final.pt \
  --output artifacts/autodl-4090/sft \
  --steps "$SFT_STEPS" --batch-size 4 --gradient-accumulation 4 \
  --max-length 768 --target-mode response_only --precision bf16 --device cuda \
  --learning-rate 2e-5 --min-learning-rate 2e-6 --warmup-steps 500 \
  --validation-fraction 0.005 --validation-every 1000 --validation-batches 50 \
  --generation-every 5000 --generation-suite data/eval/sft_generation_v1.jsonl \
  --save-every 1000 --keep-last 2 --num-workers 0 --seed 42 \
  --wandb --wandb-project MiniScale --wandb-run-name autodl4090-sft-raw \
  2>&1 | tee artifacts/autodl-4090/logs/sft.log

test -s artifacts/autodl-4090/sft/sft.pt
df -h /root/autodl-tmp
```

SFT 的 `sft.pt` 是一轮结束权重；`best_loss.pt` 和 `best_quality.pt` 分别来自固定验证 loss 和固定生成探针。原始 SFT 训练集按 conversation 去重、固定切分并全局洗牌。正式 DPO 从 `sft.pt` 开始，保持 `response_only` 与 768 context：

```bash
uv run miniscale dpo \
  --data data/raw/minimind/preference/dpo.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/autodl-4090/sft/sft.pt \
  --output artifacts/autodl-4090/dpo \
  --steps 1000 --batch-size 2 --gradient-accumulation 8 \
  --max-length 768 --target-mode response_only --precision bf16 --device cuda \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --beta 0.1 --warmup-steps 50 \
  --validation-every 100 --validation-batches 50 \
  --generation-every 500 --save-every 200 --keep-last 2 \
  --num-workers 0 --seed 42 \
  --wandb --wandb-project MiniScale --wandb-run-name autodl4090-dpo \
  2>&1 | tee artifacts/autodl-4090/logs/dpo.log

test -s artifacts/autodl-4090/dpo/dpo.pt
test -s artifacts/autodl-4090/dpo/reference.pt
```

DPO 第一步 loss 应接近 `ln(2)≈0.693`，之后观察验证 loss、reward accuracy 和 reward margin。`reference.pt` 是冻结的 SFT 参照权重，恢复 DPO 时必须一同保留。

## 6. 正式训练：数学 GRPO → 计算器 Agent RL

这里的 RL `step` 会采样 `batch_size × group_size` 条 rollout，并执行 `policy_epochs` 次优化更新；与预训练/SFT 的 step 语义不同。固定数学验证集，先检查 reward、exact match、KL 和零优势组比例；若同组总是同分，先排查输出和奖励，不要只提高学习率。

```bash
uv run miniscale grpo \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/autodl-4090/dpo/dpo.pt \
  --output artifacts/autodl-4090/grpo \
  --steps 500 --batch-size 1 --group-size 4 --policy-epochs 2 \
  --max-new-tokens 96 --precision bf16 --device cuda --reference-device same \
  --learning-rate 1e-5 --min-learning-rate 1e-6 --warmup-steps 40 \
  --temperature 1.0 --top-k 50 --beta 0.01 --clip-epsilon 0.2 \
  --validation-every 100 --validation-prompts 50 \
  --save-every 100 --keep-last 2 --seed 42 \
  --wandb --wandb-project MiniScale --wandb-run-name autodl4090-grpo \
  2>&1 | tee artifacts/autodl-4090/logs/grpo.log

test -s artifacts/autodl-4090/grpo/best.pt
```

Agent RL 继续用数学文件，因为目前只有计算器是真正可执行的工具。`best.pt` 是验证奖励选出的 GRPO 权重；如果它明显不如 `rl.pt` 的人工抽查结果，先做第 7 节的固定评估再决定父权重，并记录选择。

```bash
uv run miniscale agent-rl \
  --data data/raw/minimind/agent/agent_rl_math.jsonl \
  --tokenizer data/tokenizer/minimind \
  --checkpoint artifacts/autodl-4090/grpo/best.pt \
  --output artifacts/autodl-4090/agent-rl \
  --steps 500 --batch-size 1 --group-size 2 --policy-epochs 2 \
  --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --device cuda --reference-device same \
  --learning-rate 5e-6 --min-learning-rate 5e-7 --warmup-steps 20 \
  --temperature 1.0 --top-k 50 --beta 0.01 --clip-epsilon 0.2 \
  --validation-every 100 --validation-prompts 20 \
  --save-every 100 --keep-last 2 --seed 42 \
  --wandb --wandb-project MiniScale --wandb-run-name autodl4090-agent-rl \
  2>&1 | tee artifacts/autodl-4090/logs/agent-rl.log

test -s artifacts/autodl-4090/agent-rl/best.pt
test -s artifacts/autodl-4090/agent-rl/agent_rl.pt
```

观察 `validation_success_rate`、合法/非法工具调用率、reward 和 KL。工具调用率上涨但成功率不涨，不能单凭日志断定能力提高。

## 7. 固定评估、推理与导出

同一评估命令中比较不同阶段，避免只看各自的训练 loss。注意原生 `generate` **没有 KV cache**，长生成探针可能比预期慢；Hugging Face 标准 Llama 导出模型可以使用 Transformers 的缓存生成。

```bash
uv run miniscale evaluate-sft \
  --checkpoint artifacts/autodl-4090/sft/sft.pt \
  --checkpoint artifacts/autodl-4090/sft/best_quality.pt \
  --suite data/eval/sft_generation_v1.jsonl \
  --precision bf16 --device cuda \
  --output artifacts/autodl-4090/sft-comparison.json

uv run miniscale evaluate --kind grpo \
  --checkpoint artifacts/autodl-4090/dpo/dpo.pt \
  --checkpoint artifacts/autodl-4090/grpo/best.pt \
  --prompts 100 --max-new-tokens 96 --precision bf16 --device cuda \
  --output artifacts/autodl-4090/dpo-vs-grpo.json

uv run miniscale evaluate --kind agent \
  --checkpoint artifacts/autodl-4090/grpo/best.pt \
  --checkpoint artifacts/autodl-4090/agent-rl/best.pt \
  --prompts 50 --max-turns 3 --max-new-tokens 64 \
  --precision bf16 --device cuda \
  --output artifacts/autodl-4090/grpo-vs-agent.json

uv run miniscale generate \
  --checkpoint artifacts/autodl-4090/agent-rl/best.pt \
  --tokenizer data/tokenizer/minimind \
  --prompt '请计算 7109*2920，只给最终结果。' \
  --calculator --max-turns 3 --max-new-tokens 64 \
  --temperature 0 --device cuda
```

生成结果应有完整 transcript 和工具调用统计。训练完成并确定选用的 checkpoint 后，可导出为标准 Transformers Llama 目录；导出目录必须尚不存在：

```bash
uv run miniscale export-hf \
  --checkpoint artifacts/autodl-4090/agent-rl/best.pt \
  --tokenizer data/tokenizer/minimind \
  --output artifacts/autodl-4090/agent-rl-hf
uv run python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
p = 'artifacts/autodl-4090/agent-rl-hf'
t = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p)
print(type(m).__name__, len(t), m.config.num_hidden_layers)
PY
```

### 本地画 loss 和学习率图

每个阶段的 `*_metrics.jsonl` 都包含 `step`、`train_loss`、`learning_rate`；前三阶段的验证行另含 `validation_loss`。下面临时安装绘图依赖并在 `artifacts/autodl-4090/` 生成各阶段 PNG，**不会修改项目锁文件或训练依赖**。[uv 的临时依赖说明](https://docs.astral.sh/uv/guides/scripts/)。

```bash
uv run --with matplotlib python - <<'PY'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('artifacts/autodl-4090')
for stage in ('pretrain', 'sft', 'dpo', 'grpo', 'agent-rl'):
    name = stage.replace('-', '_')
    path = root / stage / f'{name}_metrics.jsonl'
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        continue
    steps = [row['step'] for row in rows]
    fig, (loss_ax, lr_ax) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    loss_ax.plot(steps, [row['train_loss'] for row in rows], label='train/loss')
    validation = [row for row in rows if 'validation_loss' in row]
    if validation:
        loss_ax.plot([row['step'] for row in validation],
                     [row['validation_loss'] for row in validation],
                     label='eval/loss', marker='.', linewidth=1)
    loss_ax.set_ylabel('loss')
    loss_ax.legend()
    lr_ax.plot(steps, [row['learning_rate'] for row in rows])
    lr_ax.set_ylabel('learning rate')
    lr_ax.set_xlabel('training step')
    fig.tight_layout()
    output = root / f'{stage}-loss-lr.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)
PY
```

在 AutoDL 文件浏览器打开 PNG，或下载到本机。GRPO/Agent RL 的 policy loss 不能单独代表任务质量；同时查看各自 JSONL 中的 `validation_reward`、`validation_exact_match` 或 `validation_success_rate`，或在 W&B 看对应 `eval/` 曲线。

## 8. 断线、OOM、备份与收尾

- **SSH 断线而实例仍在运行**：重新登录后 `tmux attach -t miniscale`；先检查训练是否还在，避免重复启动同一输出目录。可用 `nvidia-smi`、`tail -n 20 artifacts/autodl-4090/logs/pretrain.log` 和各阶段 `*_metrics.jsonl` 查看进度。
- **进程退出或实例重启**：列出该阶段 `checkpoints/`，挑选最新的完整周期 checkpoint；若正常 `Ctrl+C` 产生了 `emergency_step_*.pt`，也可选它。复制原始命令，保持 `--output`、数据、tokenizer、`--steps` **总步数**、batch/累计、seed、精度和其他训练轨迹参数不变：预训练加 `--resume 路径`；后四阶段把 `--checkpoint 父权重` 替换为 `--resume 路径`。不要用 smoke checkpoint 恢复正式 run，也不要只备份一个 RL/DPO checkpoint 而丢掉同目录的 `reference.pt`。详见[恢复契约](checkpointing.md)。
- **正式运行中 OOM**：恢复时不能直接改 micro-batch、group size 或 reference device。先使用最后完整 checkpoint 按原配方恢复；若原配方无法运行，用新输出目录和上一阶段权重重新开始，按第 4 节调整并重新做 smoke。恢复签名会拒绝轨迹参数变化。
- **磁盘与备份**：每阶段结束运行 `df -h /root/autodl-tmp`。把 `artifacts/autodl-4090/` 中的 run manifest、日志、评估 JSON 和选定 checkpoint 复制到已挂载的 AutoDL 文件存储或下载到本地；DPO/GRPO/Agent RL 的完整恢复还需原输出目录及 `reference.pt`。AutoDL 本地盘没有冗余，实例释放后数据不能依赖本地盘恢复。[文件存储说明](https://www.autodl.com/docs/fs/)、[实例数据说明](https://www.autodl.com/docs/instance_data/)。
- **结束租用**：确认备份可读、`git-commit.txt` 和 `download_manifest.json` 已保存，再在控制台关机。关机与释放的行为不同，按实例页面和 AutoDL 文档核对。

如果全量一轮的实测时长超出预算，可另开一个**新实验**，用 [`prepare-sft-data` 的质量策略](sft.md#训练前审计) 生成较小的派生集并重新计算步数。它只覆盖筛选后的数据，不等于全量 SFT 一轮；在实验记录中写清使用的文件和步数。
