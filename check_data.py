import pandas as pd
try:
    df = pd.read_csv('data/characteristics_data_feb2017.csv')
    print(f"Total rows: {len(df)}")
    if 'ret' in df.columns:
        nan_ret = df['ret'].isna().sum()
        print(f"NaN in ret: {nan_ret}")
        if nan_ret > 0:
            print("Warning: Target variable 'ret' contains NaNs.")
    else:
        print("Column 'ret' not found.")
except Exception as e:
    print(f"Error: {e}")
