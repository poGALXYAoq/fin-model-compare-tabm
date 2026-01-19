import pandas as pd
import numpy as np

try:
    df = pd.read_csv('data/characteristics_data_feb2017.csv')
    exclude = ['', 'yy', 'mm', 'date', 'permno', 'ret', 'prc']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features]
    print(f"Feature max values:\n{X.max().sort_values(ascending=False).head()}")
    print(f"Feature min values:\n{X.min().sort_values(ascending=True).head()}")
    
    print(f"\nExample feature 'mvel1' stats if exists:")
    if 'mvel1' in X.columns:
         print(X['mvel1'].describe())
except Exception as e:
    print(e)
