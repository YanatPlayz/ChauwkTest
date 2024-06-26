import pandas as pd 
df = pd.read_csv("./List_of_PMKKs.csv")
df.columns = df.iloc[0]
df = df[1:]
df.to_csv('updated_PMKKs.csv', index=False)