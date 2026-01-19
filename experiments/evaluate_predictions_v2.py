import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results_v2")
PRED_FILE = os.path.join(OUTPUT_DIR, "rolling_predictions.csv")

DATE_COL = "date"
TARGET_COL = "ret"
PRED_COL = "pred_ret"
NUM_QUANTILES = 5  # Group stocks into 5 buckets (Quintiles)


def evaluate_predictions():
    if not os.path.exists(PRED_FILE):
        print(f"Error: {PRED_FILE} not found. Please run the training script first.")
        return

    print(f"Loading predictions from {PRED_FILE}...")
    df = pd.read_csv(PRED_FILE)

    # Ensure date format
    if df[DATE_COL].dtype == 'object':
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Total predictions: {len(df)}")

    # --- 1. IC Analysis (Information Coefficient) ---
    print("\n--- 1. IC Analysis (Rank IC) ---")

    # Calculate Rank IC per month
    def calculate_ic(group):
        if len(group) < 10: return np.nan
        # Spearman correlation is Rank IC
        return spearmanr(group[PRED_COL], group[TARGET_COL])[0]

    ic_series = df.groupby(DATE_COL).apply(calculate_ic, include_groups=False)

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else 0

    print(f"IC Mean: {ic_mean:.4f}")
    print(f"IC Std : {ic_std:.4f}")
    print(f"IC IR  : {ic_ir:.4f}")
    print(f"Positive IC Ratio: {(ic_series > 0).mean():.2%}")

    # Plot IC Time Series
    plt.figure(figsize=(12, 6))
    plt.bar(ic_series.index, ic_series.values, color='skyblue', alpha=0.8, width=20)
    plt.axhline(ic_mean, color='red', linestyle='--', label=f'Mean IC: {ic_mean:.3f}')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Monthly Rank IC (Information Coefficient)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ic_timeseries.png'))
    plt.close()

    # --- 2. Portfolio Backtest ( quantile sorts ) ---
    print(f"\n--- 2. Portfolio Backtest ({NUM_QUANTILES} Groups) ---")

    # Function to assign quantiles per month
    def assign_quantiles(group):
        try:
            return pd.qcut(group[PRED_COL], NUM_QUANTILES, labels=False, duplicates='drop')
        except ValueError:
            # Handle cases with too few unique values
            return np.zeros(len(group))  # Assign all to group 0 if fails

    df['quantile'] = df.groupby(DATE_COL, group_keys=False).apply(assign_quantiles, include_groups=False)

    # Calculate average return for each quantile each month
    # We use equal weighted returns here for simplicity
    portfolio_rets = df.groupby([DATE_COL, 'quantile'])[TARGET_COL].mean().unstack()

    # Rename columns for clarity (0 -> Low, N -> High)
    portfolio_rets.columns = [f'G{i + 1}' for i in portfolio_rets.columns]

    # Calculate Long-Short Return (High - Low)
    portfolio_rets['Long-Short'] = portfolio_rets[f'G{NUM_QUANTILES}'] - portfolio_rets['G1']

    # Calculate Cumulative Returns
    # Using log returns for summation or simple compounding: (1+r).cumprod()
    cum_rets = (1 + portfolio_rets).cumprod()

    # Metrics for Long-Short
    ls_mean_ret = portfolio_rets['Long-Short'].mean() * 12  # Annualized (simple)
    ls_std = portfolio_rets['Long-Short'].std() * np.sqrt(12)
    ls_sharpe = ls_mean_ret / ls_std if ls_std != 0 else 0

    print(f"Long-Short Annualized Return: {ls_mean_ret:.2%}")
    print(f"Long-Short Sharpe Ratio: {ls_sharpe:.4f}")

    # Plot Cumulative Returns
    plt.figure(figsize=(12, 6))
    for col in cum_rets.columns:
        if col == 'Long-Short':
            plt.plot(cum_rets.index, cum_rets[col], label=col, linewidth=2.5, color='black')
        else:
            plt.plot(cum_rets.index, cum_rets[col], label=col, alpha=0.6)

    plt.title(f'Cumulative Returns of {NUM_QUANTILES} Portfolios (Sorted by Prediction)')
    plt.yscale('log')  # Log scale is often better for long term returns
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cumulative_returns.png'))
    plt.close()

    # Plot Monotonicity (Bar chart of average returns per group)
    avg_group_ret = portfolio_rets.drop(columns=['Long-Short']).mean() * 12  # Annualized

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("viridis", n_colors=NUM_QUANTILES)
    avg_group_ret.plot(kind='bar', color=colors)
    plt.title('Average Annualized Return per Quantile')
    plt.ylabel('Annualized Return')
    plt.xlabel('Quantile (Low Prediction -> High Prediction)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monotonicity.png'))
    plt.close()

    # --- 3. Time-Stratified Analysis ---
    print("\n--- 3. Time-Stratified Analysis ---")

    # Define periods (e.g., Decades)
    df['Year'] = df[DATE_COL].dt.year
    df['Decade'] = (df['Year'] // 10) * 10

    # Group by Decade and calculate metrics
    period_metrics = []

    for period, period_df in df.groupby('Decade'):
        # IC Analysis for period
        p_ic_series = period_df.groupby(DATE_COL).apply(calculate_ic, include_groups=False)
        p_ic_mean = p_ic_series.mean()

        # Portfolio Analysis for period
        # Re-calculate quantiles within the period to be safe, or use global ones?
        # Using already assigned 'quantile' column is fine as it was cross-sectional per month.

        p_portfolio_rets = period_df.groupby([DATE_COL, 'quantile'])[TARGET_COL].mean().unstack()
        p_portfolio_rets.columns = [f'G{i + 1}' for i in p_portfolio_rets.columns]

        if f'G{NUM_QUANTILES}' in p_portfolio_rets.columns and 'G1' in p_portfolio_rets.columns:
            p_ls_ret = p_portfolio_rets[f'G{NUM_QUANTILES}'] - p_portfolio_rets['G1']

            p_ls_ann_ret = p_ls_ret.mean() * 12
            p_ls_std = p_ls_ret.std() * np.sqrt(12)
            p_ls_sharpe = p_ls_ann_ret / p_ls_std if p_ls_std != 0 else 0
        else:
            p_ls_ann_ret = np.nan
            p_ls_sharpe = np.nan

        period_metrics.append({
            'Period': f"{period}s",
            'IC Mean': p_ic_mean,
            'L-S Return': p_ls_ann_ret,
            'Sharpe': p_ls_sharpe,
            'Months': len(period_df[DATE_COL].unique())
        })

    metrics_df = pd.DataFrame(period_metrics)
    print("\nPerformance by Decade:")
    print(metrics_df.to_string(index=False, float_format="%.4f"))

    # Plot Sub-period Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = metrics_df['Period']
    ax1.bar(x, metrics_df['IC Mean'], color='skyblue', alpha=0.7, label='IC Mean')
    ax1.set_ylabel('IC Mean', color='blue')
    ax1.set_ylim(0, max(metrics_df['IC Mean']) * 1.2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(x, metrics_df['Sharpe'], color='red', marker='o', linewidth=2, label='Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Model Performance by Decade')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'subperiod_metrics.png'))
    plt.close()

    print("\nVisualization saved to experiments/results/:")
    print("- ic_timeseries.png")
    print("- cumulative_returns.png")
    print("- monotonicity.png")
    print("- subperiod_metrics.png")


if __name__ == "__main__":
    evaluate_predictions()

