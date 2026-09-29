# 训练正确性与系统优化（2026-09-29）

范围为 `rebuild/`，保留原有训练阶段、API、checkpoint 格式 v2 和轻量 PyTorch 实现。
审查覆盖模型、tokenizer、数据下载/清洗/审计、全部训练阶段、恢复、推理、追踪、CLI、配置、文档和原有 113 项测试。

## 已确认并修复的问题

| 问题 | 修改及原因 |
| --- | --- |
| autocast 下 residual 为 FP32，RoPE 因而把 BF16 Q/K 提升到 FP32 | 使用投影的 dtype；测试直接检查 SDPA 的 Q/K/V dtype，并执行 CPU/CUDA backward |
| 完整 checkpoint 在推理/恢复时直接加载到 GPU，包含多余 optimizer/model 副本 | 统一 CPU 反序列化，恢复时仅将实际参数和 optimizer 状态迁移到目标设备；模型加载不消耗调用者 CPU RNG |
| 中途 forward/backward 消耗 RNG，紧急 checkpoint 却记录上一次完整更新的数据位置 | 保存更新开始的 RNG，未修改 optimizer 时回退 RNG；带 dropout、gradient accumulation 的 Pretrain/SFT/DPO 恢复逐张量一致 |
| optimizer 中断、RL 多次 policy epoch 中断可能保存“半步”参数；RL 提前推进任务游标 | 跟踪更新边界；开始修改参数后若失败，不写伪完整 emergency；任务游标在整组更新成功后推进 |
| RL rollout/old-policy 使用 eval，而 policy update 启用 dropout | policy update 保持 eval 并正常求梯度，避免初始 ratio/KL 带有 dropout 噪声 |
| 多 worker 预训练的新 epoch 重置局部 iteration，shuffle 重复 | 使用 DataLoader 生成的递进 worker seed；测试两个 epoch 顺序不同且可重放、跨 epoch resume 一致 |
| 独立 validation 文件存在时，预训练仍从训练文件扣除内部 hash 验证集 | 使用完整训练文件，保持独立验证文件职责明确 |
| 索引数据集在父进程先读后 fork 时共享文件偏移 | 按进程 ID 重新打开文件；SFT/DPO 多 worker 内容与单进程一致 |
| 计算器推理忽略 seed、top-p 和重复抑制参数 | 将同一个私有 generator 和解码参数传入每轮工具 rollout |
| SFT best-loss 在同一步 generation 之前保存，early stopping 状态落后 | 在质量计数更新后保存 best-loss；比对同一步 best-loss/best-quality 状态 |
| resume 后 W&B 可能补传恢复点以后的旧事件 | 连接前截断本地待传队列；结束时刷新队列。已上传云端历史不能由本地撤销 |
| 没有 HF 模型导出，只有 HF tokenizer | 增加 `export-hf`，映射到标准 Llama；保留 tied embeddings、GQA、RoPE、SwiGLU、特殊 ID 和 tokenizer；原生训练代码无需继承 HF Trainer |

模型配置现在提前拒绝零 head、奇数 RoPE head dimension、越界 token ID 和非法数值。
全 mask 标签给出明确错误。生成只对最后一个位置计算词表 logits，采样概率使用 FP32；
固定长度预训练 block 使用 SDPA 的原生 causal 路径，避免额外构造方形 mask。

保留了已有正确设计：单次 next-token shift、assistant/action mask、SFT 按监督 token 加权、
DPO 按 pair 加权、冻结 reference、固定验证抽样、原子 checkpoint、AdamW 分组和按更新调度 LR。
没有增加通用 Trainer、多卡框架或新的运行依赖。

## 兼容性与恢复边界

五个生产阶段的 implementation version 提升至 3，因为 BF16 数值、采样精度、数据顺序及 RL dropout
语义会影响训练轨迹。旧权重仍可推理、导出和阶段衔接；旧 implementation v2 的严格 resume 会被拒绝。
继续旧实验应使用旧代码，不能通过隐式版本迁移宣称精确恢复。详见 [checkpointing.md](checkpointing.md)。

optimizer 开始修改参数后、验证和状态更新完成前的异常需要回到最近完整周期/best checkpoint；没有可用完整 checkpoint 时需重启
该实验。正常 forward/backward 阶段的紧急保存不包含部分梯度，恢复会重做尚未完成的更新。
没有为每个 step 复制整份 AdamW，以免增加单卡训练开销。

## 验证

运行环境：Python 3.12.3、PyTorch 2.9.1+cu128、CUDA 12.8、RTX 5060 Laptop GPU（8GB）。
各阶段 run manifest 新增版本、设备、线程数、TF32 与 deterministic algorithms 状态。

```bash
cd rebuild
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  ../.venv/bin/python -m unittest discover -s tests -q
```

- 修改前：113 项测试通过。
- 修改后：133 项测试全部通过（新增 20 项）；覆盖恢复、worker、BF16、HF 数值等价、生成与追踪回归。
- 原有 smoke pipeline 与 JSONL 五阶段集成测试通过。
- 真实 MiniMind 各源取前 64 行，使用词表 6400、216,416 参数微型模型运行 CUDA BF16：
  Pretrain/SFT/DPO 各 2 step，GRPO/Agent RL 各 1 step、每 step 2 policy epochs。所有 loss 有限。
- CUDA 预训练从 step 1 恢复到 step 2，最终权重逐张量相同。
- HF 导出经 `AutoModelForCausalLM` / `AutoTokenizer` 离线加载，FP32 logits 最大绝对误差为 0；
  单元测试另覆盖 loss、padding、tied weights 和带 KV cache 的 greedy generation。
- 本地报告：`artifacts/systems-review-20260929/result.json`。该目录被 Git 忽略。

微型 RL smoke 的 reward 为 0，不证明学到策略；它验证采样、目标计算、冻结 reference、更新、验证和
checkpoint 能衔接。未执行 64M 全量训练，也未据此宣称吞吐或收敛质量提升。

## 后续优先项

1. 固定 64M 单卡训练配方，做 BF16 长跑、吞吐/显存基准及验证质量对照。
2. 预 tokenization 缓存和可恢复数据游标，减少每次 resume 重放全部已消费数据的成本。
3. 增加 prompt/近重复去重与 train/validation 交集检查；审查 SFT 模板中正文包含角色分隔符的边界。
4. 补充独立通用聊天、代码执行和数学评测；以 RL 有效非零 reward 与失败案例衡量训练收益。
5. 评估原生 KV cache、activation checkpointing 和 GQA backend，逐项记录速度、显存与数值对照。
