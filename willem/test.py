import pandas as pd
df = pd.read_csv("evaluation/results/solution.csv")
print(df.iloc[11185:18685].label.value_counts())   # should be H 4259 / A 2506 / D 735