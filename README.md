# fin-model-compare-tabm
本项目利用 ICLR 2025 论文提到的 TabM (Efficient Ensembling of MLPs) 模型，对资产收益率（returns）进行滚动窗口预测研究。
# Asset Return Prediction via TabM

## 核心特性
- **模型**: 基于 `tabm` 库的参数高效集成模型（TabM-BatchEnsemble）。
- **策略**: 采用滚动窗口（Rolling Window）训练预测逻辑，防止 Look-ahead bias。
- **特征**: 支持连续特征（分段线性嵌入）和类别特征。

## 项目结构
```text
├── data/               # 原始 CSV 数据
├── src/
│   ├── data_utils.py   # 数据清洗、特征标准化、滚动窗口切分
│   ├── model_factory.py# TabM 模型定义与 PLE 嵌入配置
│   └── trainer.py      # 训练循环、GPU 调度与早停逻辑
├── experiments/
│   └── run_rolling.py  # 滚动实验主脚本
├── pyproject.toml      # uv 配置文件 (含 CUDA 镜像源)
└── .gitignore
```


📈 滚动预测逻辑 (Rolling Window)
项目采用 12+1 滚动模式：

训练期: 过去 12 个月的截面数据。

预测期: 第 13 个月的所有资产收益率。

步进: 每次向后滑动 1 个月，严格模拟实盘样本外表现，防止 Look-ahead bias。

🧠 模型特性
Piecewise Linear Embeddings (PLE): 自动对金融指标进行非线性分桶处理，相比直接输入原始数值，能显著提升深度学习在表格数据上的泛化性。

Parameter-Efficient Ensembling: 仅需训练一个模型的时间，即可获得 32 个子模型的集成效果。

🚀 运行实验
准备数据: 将 CSV 数据放入 data/ 目录。确保包含 date, permno, ret 以及其他数值特征列。

## 🚀 运行实验

在项目根目录下执行，确保 Python 能够识别 `src` 模块：

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
uv run python experiments/run_rolling.py

# Linux / macOS
PYTHONPATH=. uv run python experiments/run_rolling.py
```

启动训练:

`uv run python experiments/run_rolling.py`

查看结果: 结果将自动保存至 experiments/rolling_predictions.csv，包含日期、资产 ID、真实收益率及预测收益率。