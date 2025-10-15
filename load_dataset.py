import pandas as pd

print("Loading dataset...")

df = pd.read_csv("ecommerce_purchases_1000.csv")
print(" Dataset loaded successfully!")

print(df.head())
