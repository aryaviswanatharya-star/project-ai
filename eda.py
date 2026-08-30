import pandas as pd

# Load datasets
mal = pd.read_csv("malayalam_movies.csv")
tam = pd.read_csv("tamil_movies.csv")

print("========== Malayalam Dataset Information ==========")
mal.info()

print("\n========== Tamil Dataset Information ==========")
tam.info()

print("\n========== Dataset Shape ==========")
print("Malayalam:", mal.shape)
print("Tamil:", tam.shape)

print("\n========== Malayalam Columns ==========")
print(mal.columns)

print("\n========== Tamil Columns ==========")
print(tam.columns)

print("\n========== First Five Malayalam Movies ==========")
print(mal.head())

print("\n========== First Five Tamil Movies ==========")
print(tam.head())

print("\n========== Malayalam Missing Values ==========")
print(mal.isnull().sum())

print("\n========== Tamil Missing Values ==========")
print(tam.isnull().sum())