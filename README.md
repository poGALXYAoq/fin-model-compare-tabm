# fin-model-compare-tabm
本项目利用 ICLR 2025 论文提到的 TabM (Efficient Ensembling of MLPs) 模型，对资产收益率（returns）进行滚动窗口预测研究。
# Asset Return Prediction via TabM

## 核心特性
- **模型**: 基于 `tabm` 库的参数高效集成模型（TabM-BatchEnsemble）。
- **策略**: 采用滚动窗口（Rolling Window）训练预测逻辑，防止 Look-ahead bias。
- **特征**: 支持连续特征（分段线性嵌入）和类别特征。

## 项目结构
```text
├── data/               # 存放原始 CSV 数据
├── src/
│   ├── data_utils.py   # 数据加载、预处理与滚动切片
│   ├── model_factory.py# TabM 模型定义与配置
│   └── trainer.py      # 训练循环与早停逻辑
├── experiments/
│   └── run_rolling.py  # 滚动实验主脚本
├── requirements.txt    # 依赖项
└── .gitignore
