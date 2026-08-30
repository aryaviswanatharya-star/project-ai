import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
mal = pd.read_csv("malayalam_movies_final_2.csv")
tam = pd.read_csv("Tamil_movies_dataset.csv")

# Display first 5 rows
print("========== Malayalam Dataset ==========")
print(mal.head())

print("\n========== Tamil Dataset ==========")
print(tam.head())

# Display column names
print("\n========== Malayalam Columns ==========")
print(mal.columns)

print("\n========== Tamil Columns ==========")
print(tam.columns)

# Check missing values
print("\n========== Malayalam Missing Values ==========")
print(mal.isnull().sum())

print("\n========== Tamil Missing Values ==========")
print(tam.isnull().sum())

# Remove missing values
mal = mal.dropna()
tam = tam.dropna()

# Display dataset size
print("\n========== Malayalam Shape ==========")
print(mal.shape)

print("\n========== Tamil Shape ==========")
print(tam.shape)

# Display dataset information
print("\n========== Malayalam Info ==========")
mal.info()

print("\n========== Tamil Info ==========")
print(tam.info())
print("Malayalam Columns")
print(mal.columns)

print("\nTamil Columns")
print(tam.columns)
# Select important columns from Malayalam dataset

mal = mal[['title',
           'genres',
           'keywords',
           'cast',
           'directors',
           'overview']]

print("\nMalayalam Dataset")
print(mal.head())
# Select important columns from Tamil dataset

tam = tam[['MovieName',
           'Genre',
           'Director',
           'Actor']]

print("\nTamil Dataset")
print(tam.head())
# Replace missing values

mal.fillna('', inplace=True)
tam.fillna('', inplace=True)

print("\nMissing values removed successfully.")
mal['tags'] = (
    mal['title'] + " " +
    mal['genres'] + " " +
    mal['keywords'] + " " +
    mal['cast'] + " " +
    mal['directors'] + " " +
    mal['overview']
)
tam['tags'] = (
    tam['MovieName'] + " " +
    tam['Genre'] + " " +
    tam['Director'] + " " +
    tam['Actor']
)
print("\nMalayalam Tags")
print(mal[['title', 'tags']].head())

print("\nTamil Tags")
print(tam[['MovieName', 'tags']].head())