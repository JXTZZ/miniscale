# 项目结构

MiniScale 采用“稳定公开入口 + 小型基础模块 + 分阶段训练实现”的结构。重构时优先保持
数据格式、checkpoint、CLI 和 Python 导入路径兼容，不使用通用 `utils.py` 汇集无关逻辑。

```text
src/miniscale/
├── __main__.py            # python -m miniscale 入口，仅转交 CLI
├── cli/
│   ├── __init__.py        # 命令执行与稳定的程序化 CLI API
│   └── parser.py          # argparse 参数和默认值映射
├── config.py              # 模型结构配置
├── model.py               # Transformer 模型实现
├── data/
│   ├── __init__.py        # 预训练数据集与通用批处理
│   ├── sft.py 等          # 各阶段数据格式、索引和切分
│   └── *_audit.py         # 数据审计与派生数据准备
├── integrity.py           # 内容身份与安全原子写入
└── training/
    ├── core/              # runtime、checkpoint、artifacts 与兼容工具
    ├── configs/           # 各阶段配置及恢复格式常量
    ├── objectives/        # DPO/GRPO 等纯目标函数
    ├── evaluators/        # 阶段验证与生成评估
    └── stages/            # Pretrain/SFT/DPO/GRPO/Agent-RL 工作流
```

测试同样按职责分组：

```text
tests/
├── contracts/            # 公开 API 和旧路径兼容
├── core/                 # 模型与 tokenizer
├── data/                 # 数据格式、切分和审计
├── training/             # 训练目标、阶段与恢复
└── runtime/              # 推理、流水线、评估和 tracking
```

## 依赖与实现原则

- 标准库或现有依赖能保持语义时直接复用。例如变长 batch 使用 PyTorch
  `pad_sequence`，文件替换使用 `Path.replace`。
- 若库函数会改变既有行为，则保留项目实现并集中维护。例如审计百分位沿用整数最近索引，
  避免 NumPy/`statistics.quantiles` 的插值差异；checkpoint 继续保存项目特有的训练、随机数
  和数据身份状态。
- `miniscale.cli`、`miniscale.data`、旧的 `miniscale.sft_data`、
  `miniscale.training.common` 等路径仍是兼容入口。新代码应优先依赖 `data/` 和
  `training/` 下按职责划分的新模块。
- 训练阶段可以共享运行时和产物设施，但不通过通用 Trainer 隐藏 DPO、GRPO、Agent-RL 的
  不同采样、目标函数和恢复语义。

## 修改边界

新增训练能力时，应把纯目标函数、配置、评估和阶段编排分别放入现有同类模块。任何改变
checkpoint schema、数据切分或 CLI 参数的修改，都应先增加兼容测试或显式提升格式版本。

## `__main__.py` 的作用

Python 执行 `python -m miniscale doctor` 时会寻找 `miniscale/__main__.py`。该文件只导入并调用
CLI 的 `main` 函数，使模块启动方式与安装后的 `miniscale doctor` 命令行为一致。删除它不会影响
普通 `import miniscale`，但会让 `python -m miniscale` 无法运行，因此保留为三行左右的标准入口。
