import torch
import pandas as pd
import numpy as np
from src.model_factory import create_tabm_model, compute_tabm_loss
from src.data_utils import prepare_rolling_data
# 移除了 preprocess_features，改为手动处理
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import EarlyStopping
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. 配置与手动特征列表 (白名单) ---
# 必须与 TabPFN 完全一致，确保公平对比
MANUAL_FEATURE_LIST = [
    # 动量类 (Momentum)
    "cum_return_1_0", "cum_return_12_2", "cum_return_12_7", "cum_return_36_13",

    # 估值与基本面 (Valuation & Fundamentals)
    "a2me", "ato", "beme", "c", "cto", "d2a", "dpi2a", "e2p",
    "fc2y", "free_cf", "investment", "noa", "oa", "ol",
    "pcm", "pm", "prof", "q", "rna", "roa", "roe",
    "s2p", "sga2m", "at", "lev",

    # 市场属性 (Market based - lagged)
    "lme", "lturnover", "beta", "idio_vol", "spread_mean",
    "suv", "rel_to_high_price"
]

TARGET_COL = 'ret'
DATE_COL = 'date'

# --- 2. 加载与清洗数据 ---
try:
    print("Loading data...")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "characteristics_data_feb2017.csv")
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("Error: Data file not found in data/ subdirectory.")
    exit(1)

# 确保日期格式正确并排序 (非常重要，否则滚动窗口会错乱)
if df[DATE_COL].dtype == 'object':
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# 检查特征是否存在
available_features = [c for c in MANUAL_FEATURE_LIST if c in df.columns]
if len(available_features) != len(MANUAL_FEATURE_LIST):
    missing = set(MANUAL_FEATURE_LIST) - set(available_features)
    print(f"Warning: Missing features: {missing}")
feature_cols = available_features

print(f"Selected {len(feature_cols)} features for training.")

# --- 3. 准备数据矩阵 (X, y) ---
# 深度学习模型非常讨厌 NaN，必须填充。
# 既然跳过了 preprocess_features，这里必须手动处理 NaN。通常填 0 即可。
print("Handling NaN values (filling with 0)...")
df[feature_cols] = df[feature_cols].fillna(0.0)

# 提取 X 和 y
X = df[feature_cols].values
y = df[TARGET_COL].values

# 生成滚动窗口索引
# 注意：prepare_rolling_data 依赖 df 的日期列，所以一定要先排序 df
windows = prepare_rolling_data(df, train_months=12)

results = []
BATCH_SIZE = 16384

# --- 4. 滚动训练 ---
print(f"Starting rolling training with {len(windows)} windows...")

for i, (train_idx, test_idx, test_month) in enumerate(windows):
    print(f"\n[Window {i + 1}/{len(windows)}] Testing on {test_month}")

    # 进一步切分：用训练窗口的最后 10% 做验证
    split_point = int(len(train_idx) * 0.9)
    actual_train_idx = train_idx[:split_point]
    val_idx = train_idx[split_point:]

    # --- 样本量熔断机制 ---
    MIN_SAMPLES = 500
    if len(actual_train_idx) < MIN_SAMPLES:
        print(f"  [Skipping] Not enough training samples ({len(actual_train_idx)} < {MIN_SAMPLES})")
        continue

    # --- 数据提取与预处理 (标准化) ---
    # 1. 提取原始 numpy 数组
    X_train_raw = X[actual_train_idx]
    y_train_raw = y[actual_train_idx]
    X_val_raw = X[val_idx]
    y_val_raw = y[val_idx]
    X_test_raw = X[test_idx]

    # 2. 标准化 (Fit on TRAIN ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)  # 验证集用训练集的参数缩放
    X_test_scaled = scaler.transform(X_test_raw)  # 测试集用训练集的参数缩放

    # 3. 转换为 Tensor
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    # [修复点]: 必须 reshape 为 (N, 1)，否则 compute_tabm_loss 会报错
    y_train_t = torch.tensor(y_train_raw, dtype=torch.float32).reshape(-1, 1)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    # [修复点]: 验证集标签也需要 reshape
    y_val_t = torch.tensor(y_val_raw, dtype=torch.float32).reshape(-1, 1).to(device)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # 准备 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    # 注意：init_sample_size 用来初始化 TabM 的 Quantization Binning
    init_sample_size = min(20000, len(X_train_t))
    model = create_tabm_model(X_train_t[:init_sample_size].to(device), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=5, verbose=False)

    # 训练循环
    pbar = tqdm(range(50), desc="Epochs", leave=False)
    for epoch in pbar:
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = compute_tabm_loss(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = compute_tabm_loss(val_output, y_val_t)

        pbar.set_postfix({'val_loss': f'{val_loss.item():.6f}'})

        # 检查早停
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            break

    # 加载表现最好的模型进行最终预测
    model = early_stopping.load_best_model(model)

    # 3. 预测
    model.eval()
    with torch.no_grad():
        y_pred_ensemble = model(X_test_t).mean(dim=1).cpu().numpy()

    # 保存结果
    res = df.loc[test_idx, ['date', 'permno', 'ret']].copy()
    res['pred_ret'] = y_pred_ensemble
    results.append(res)

    # --- 显存清理 ---
    del model, optimizer, early_stopping
    del X_train_t, y_train_t, X_val_t, y_val_t, X_test_t
    torch.cuda.empty_cache()

# 合并并保存结果
if results:
    final_res = pd.concat(results)

    # 修复: 使用绝对路径构建 output_path，并确保目录存在
    output_dir = os.path.join(PROJECT_ROOT, 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'rolling_predictions.csv')
    final_res.to_csv(output_path, index=False)
    print(f"Rolling experiment completed! Saved to {output_path}")
else:
    print("No results generated.")
