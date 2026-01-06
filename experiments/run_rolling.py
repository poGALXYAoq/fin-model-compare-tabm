import torch
import pandas as pd
from src.model_factory import create_tabm_model, compute_tabm_loss
from src.data_utils import prepare_rolling_data, preprocess_features
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import EarlyStopping

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载数据
df = pd.read_csv('data/characteristics_data_feb2017.csv')
X, y, feature_names = preprocess_features(df)
windows = prepare_rolling_data(df, train_months=12)

results = []

# 2. 滚动训练
for train_idx, test_idx, test_month in windows:
    # 进一步切分：用训练窗口的最后一个月做验证
    # 这里假设 train_idx 是按时间排序的
    split_point = int(len(train_idx) * 0.9) 
    actual_train_idx = train_idx[:split_point]
    val_idx = train_idx[split_point:]

    X_train_t = torch.tensor(X[actual_train_idx], dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y[actual_train_idx], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32).to(device)

    # 初始化模型（使用训练集统计特性初始化 PLE）
    model = create_tabm_model(X_train_t, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    
    early_stopping = EarlyStopping(patience=5, verbose=False)

    # 训练循环
    for epoch in range(100): # 最大 epoch 可以设大一点
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = compute_tabm_loss(output, y_train_t)
        loss.backward()
        optimizer.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = compute_tabm_loss(val_output, y_val_t)
        
        # 检查早停
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 加载表现最好的模型进行最终预测
    model = early_stopping.load_best_model(model)
    
    # 3. 预测
    model.eval()
    with torch.no_grad():
        # 推理时平均 k 个预测结果
        y_pred_ensemble = model(X_test).mean(dim=1).cpu().numpy()
        
    # 保存结果
    res = df.loc[test_idx, ['date', 'permno', 'ret']].copy()
    res['pred_ret'] = y_pred_ensemble
    results.append(res)

# 合并并保存结果
final_res = pd.concat(results)
final_res.to_csv('experiments/rolling_predictions.csv', index=False)
print("Rolling experiment completed!")