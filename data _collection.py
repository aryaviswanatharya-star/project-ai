import pandas as pd
import numpy as np

# Load Malayalam Movie Dataset
mal = pd.read_csv("malayalam_movies.csv")

# Load Tamil Movie Dataset
tam = pd.read_csv("tamil_movies.csv")

print("========== Malayalam Dataset ==========")
print(mal.head())

print("\n========== Tamil Dataset ==========")
print(tam.head())
