import torch
import pandas as pd
import numpy as np
from src.model_factory import create_tabm_model, compute_tabm_loss
from src.data_utils import prepare_rolling_data, preprocess_features
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import EarlyStopping
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载数据
try:
    print("Loading data...")
    df = pd.read_csv('data/characteristics_data_feb2017.csv')
except FileNotFoundError:
    print("Error: Data file not found in data/ subdirectory.")
    exit(1)

# 注意：preprocess_features 现在会返回排序后的 df
X, y, feature_names, df = preprocess_features(df)
windows = prepare_rolling_data(df, train_months=12)

results = []
BATCH_SIZE = 16384

# 3. 滚动训练
print(f"Starting rolling training with {len(windows)} windows...")

# 引入标准化工具
from sklearn.preprocessing import StandardScaler

for i, (train_idx, test_idx, test_month) in enumerate(windows):
    print(f"\n[Window {i+1}/{len(windows)}] Testing on {test_month}")
    
    # 进一步切分：用训练窗口的最后 10% 做验证
    split_point = int(len(train_idx) * 0.9) 
    actual_train_idx = train_idx[:split_point]
    val_idx = train_idx[split_point:]

    # --- 样本量熔断机制 ---
    # 既然 n_bins 固定为 48，我们需要保证样本量足够大以计算分位数。
    # 如果 12 个月的训练数据少于 500 条 (平均每月 < 42 条)，则认为数据不足，跳过训练。
    MIN_SAMPLES = 500 
    if len(actual_train_idx) < MIN_SAMPLES:
        print(f"  [Skipping] Not enough training samples ({len(actual_train_idx)} < {MIN_SAMPLES}) for stable PLE binning.")
        continue

    # --- 数据提取与预处理 (标准化) ---
    # 1. 提取原始 numpy 数组
    X_train_raw = X[actual_train_idx]
    y_train_raw = y[actual_train_idx]
    X_val_raw = X[val_idx]
    y_val_raw = y[val_idx]
    X_test_raw = X[test_idx]

    # 2. 标准化 (Fit on TRAIN ONLY)
    # 这也能解决数值过大导致的 loss 爆炸和训练缓慢问题
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 3. 转换为 Tensor
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_raw, dtype=torch.float32) # y 通常不需要标准化，除非分布极其异常
    
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_raw, dtype=torch.float32).to(device)
    
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # 准备 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型（使用部分训练数据计算统计量）
    init_sample_size = min(20000, len(X_train_t))
    # 注意：传入的是已经标准化的数据
    model = create_tabm_model(X_train_t[:init_sample_size].to(device), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=5, verbose=False) # 设为 False 减少刷屏

    # 训练循环
    pbar = tqdm(range(50), desc="Epochs", leave=False) # leave=False 避免进度条刷屏
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
        
        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = compute_tabm_loss(val_output, y_val_t)
        
        pbar.set_postfix({'val_loss': f'{val_loss.item():.6f}'})

        # 检查早停
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            # print(f"  Early stopping at epoch {epoch}")
            break
    
    # 加载表现最好的模型进行最终预测
    model = early_stopping.load_best_model(model)
    
    # 3. 预测
    model.eval()
    with torch.no_grad():
        # 推理时平均 k 个预测结果
        y_pred_ensemble = model(X_test_t).mean(dim=1).cpu().numpy()
    
    # 保存结果
    res = df.loc[test_idx, ['date', 'permno', 'ret']].copy()
    res['pred_ret'] = y_pred_ensemble
    results.append(res)

    # --- 显存清理 ---
    # Python 的垃圾回收对 GPU 显存释放不一定及时，手动清理是长序列训练的最佳实践
    del model, optimizer, early_stopping
    del X_train_t, y_train_t, X_val_t, y_val_t, X_test_t
    torch.cuda.empty_cache() # 强制释放显存碎片

# 合并并保存结果
if results:
    final_res = pd.concat(results)
    output_path = 'experiments/rolling_predictions.csv'
    final_res.to_csv(output_path, index=False)
    print(f"Rolling experiment completed! Saved to {output_path}")
else:
    print("No results generated. Check data path or rolling window logic.")