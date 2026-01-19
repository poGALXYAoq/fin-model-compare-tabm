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
    # 排除 ID 和目标变量
    exclude = ['', 'yy', 'mm', 'date', 'permno', 'ret', 'prc']
    features = [c for c in df.columns if c not in exclude]

    # 1. 强制排序：时间优先，其次是个股 ID
    # 这是滚动窗口预测能在正确时间点切分的关键
    df = df.sort_values(['date', 'permno']).reset_index(drop=True)
    
    # 2. 缺失值处理
    # 金融数据中 NaN 很常见。
    # 简单方案：填 0 (假设该因子处于平均水平或无信号)
    # 进阶方案(TODO)：使用截面中位数填充 df.groupby('date')[features].transform(lambda x: x.fillna(x.median()))
    X = df[features].fillna(0).values 
    y = df['ret'].values.reshape(-1, 1)
    
    return X, y, features, df # 返回排序后的 df 以便后续索引对齐