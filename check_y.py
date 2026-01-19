import pandas as pd
try:
    df = pd.read_csv('data/characteristics_data_feb2017.csv')
    print("Statistics for 'ret' column:")
    print(df['ret'].describe())
    print("\nCheck for logic errors in ret:")
    print(f"Max ret: {df['ret'].max()}")
    print(f"Min ret: {df['ret'].min()}")
    print(f"Std dev: {df['ret'].std()}")
except Exception as e:
    print(e)
