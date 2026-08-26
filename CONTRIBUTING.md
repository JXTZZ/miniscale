# Contributing to MiniScale

感谢你改进 MiniScale。本项目强调“代码小而语义完整”：改动应尽量保持训练目标清晰、实验可复现、
checkpoint 可解释，并用小模型测试验证算法语义。

## 开发环境

项目支持 Python 3.11 和 3.12，推荐使用仓库固定的 Python 3.12：

```bash
uv sync
uv run miniscale doctor
uv run python -m unittest discover -s tests -v
```

真实数据、模型权重、W&B 离线文件和 `artifacts/` 不应提交。测试应使用临时目录、smoke 配置和最小
合成数据，不能依赖本地 `data/` 内容、GPU、网络或外部账号。

## 代码边界

- `src/miniscale/` 是可安装的核心实现；`trainer/` 只保留便于阅读的薄 smoke 入口。
- 正式预训练入口是 `miniscale pretrain` / `run_pretrain_jsonl`；`run_pretrain` 只用于秒级集成测试。
- 生产预训练参数的默认值只定义在 `PretrainOptions`，CLI 通过 `pretrain_option_default` 读取。
- 模型配置属于 checkpoint 格式的一部分；加载 checkpoint 时不要用隐式默认值替换保存的配置。
- 数据顺序、优化器、精度、scheduler、初始化或 loss 语义的变化必须进入 resume identity；必要时提升
  implementation/signature/checkpoint 版本并补迁移说明。
- checkpoint 必须先完成结构与兼容性校验，再修改模型、优化器或随机状态。

## 测试要求

按改动风险选择最小测试，并在提交前运行完整测试：

```bash
# 单文件快速反馈
uv run python -m unittest tests.test_pretrain -v

# 合入前完整回归
uv run python -m unittest discover -s tests -v
```

训练代码至少应覆盖：一次 forward/backward、非有限值失败、产物格式，以及 uninterrupted/resume 的
结果一致性。修改 packing、mask、shift、reward 或 advantage 时，应写一个能手工推导期望值的小例子，
不要只断言“程序能运行”。

## Pull request 检查表

- [ ] 改动聚焦，未混入 checkpoint、数据或无关格式化。
- [ ] 用户可见的 CLI、产物或训练语义变化已更新 README 或 `docs/`。
- [ ] 新配置有输入校验，并记录进 manifest/checkpoint（如会影响训练轨迹）。
- [ ] 新数据处理说明了确定性、worker 行为、内存复杂度和 train/validation 边界。
- [ ] 完整测试通过；无法运行的硬件专项测试已在 PR 中说明。
- [ ] 没有把 smoke 默认值包装成看似正式的训练 recipe。

公开发布前还需要由仓库所有者明确选择并添加许可证；贡献者不应自行猜测许可类型。
