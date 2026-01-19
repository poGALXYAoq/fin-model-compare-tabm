import pandas as pd
import numpy as np
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_metrics(df):
    """
    计算 Rank IC, Rank ICIR 以及简单的多空收益。
    假设 df 包含: date, permno, ret, pred_ret
    """
    # 确保日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. 计算 Rank IC (每天计算 Spearman 相关系数)
    # pandas 的 corr(method='spearman') 等价于 rank 后计算 pearson
    ic_series = df.groupby('date')[['pred_ret', 'ret']].apply(
        lambda x: x['pred_ret'].corr(x['ret'], method='spearman'),
        include_groups=False
    )
    
    rank_ic = ic_series.mean()
    rank_icir = rank_ic / ic_series.std()
    
    print(f"Daily Rank IC Stats:")
    print(f"  Mean: {rank_ic:.4f}")
    print(f"  Std : {ic_series.std():.4f}")
    print(f"  ICIR: {rank_icir:.4f}")
    
    # 2. 多空组合收益 (Long-Short Return)
    # 简单策略：每天做多预测最高的 10%，做空预测最低的 10%
    def calc_ls_ret(x):
        q_high = x['pred_ret'].quantile(0.9)
        q_low = x['pred_ret'].quantile(0.1)
        
        long_ret = x[x['pred_ret'] >= q_high]['ret'].mean()
        short_ret = x[x['pred_ret'] <= q_low]['ret'].mean()
        
        return long_ret - short_ret

    ls_series = df.groupby('date').apply(calc_ls_ret)
    avg_ls_ret = ls_series.mean()
    annualized_ls_ret = avg_ls_ret * 12 # 假设月频数据
    
    print(f"\nLong-Short Portfolio Stats:")
    print(f"  Avg Monthly Return: {avg_ls_ret:.4f}")
    print(f"  Annualized Return : {annualized_ls_ret:.4f}")

    return ic_series, ls_series

if __name__ == "__main__":
    try:
        df = pd.read_csv('experiments/rolling_predictions.csv')
        print("Loaded experiments/rolling_predictions.csv")
        compute_metrics(df)
    except FileNotFoundError:
        print("Prediction file not found. Please run experiments/run_rolling.py first.")
