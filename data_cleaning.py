import pandas as pd

# Load datasets
mal = pd.read_csv("malayalam_movies.csv")
tam = pd.read_csv("tamil_movies.csv")

# Check missing values
print("Malayalam Missing Values")
print(mal.isnull().sum())

print("\nTamil Missing Values")
print(tam.isnull().sum())

# Remove missing values
mal = mal.dropna()
tam = tam.dropna()

# Check duplicate records
print("\nMalayalam Duplicate Records:", mal.duplicated().sum())
print("Tamil Duplicate Records:", tam.duplicated().sum())

# Remove duplicate records
mal = mal.drop_duplicates()
tam = tam.drop_duplicates()

# Check data types
print("\n========== Malayalam Info ==========")
mal.info()

print("\n========== Tamil Info ==========")
tam.info()

print("\nData Cleaning Completed Successfully")
