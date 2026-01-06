import pandas as pd
import numpy as np

def prepare_rolling_data(df, train_months=12):
    """
    生成滚动窗口的索引
    """
    df['date'] = pd.to_datetime(df['date'])
    unique_months = sorted(df['date'].unique())
    
    windows = []
    for i in range(len(unique_months) - train_months):
        train_period = unique_months[i : i + train_months]
        test_period = unique_months[i + train_months] # 第13个月
        
        train_idx = df[df['date'].isin(train_period)].index
        test_idx = df[df['date'] == test_period].index
        
        if len(test_idx) > 0:
            windows.append((train_idx, test_idx, test_period))
            
    return windows

def preprocess_features(df):
    """
    特征选择与标准化
    """
    # 排除 ID 和目标变量
    exclude = ['', 'yy', 'mm', 'date', 'permno', 'ret', 'prc']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features].fillna(0).values # 简单填充，金融建议用更复杂的
    y = df['ret'].values.reshape(-1, 1)
    
    return X, y, features