import pandas as pd

df = pd.read_csv('hotel_reviews_kaggle.csv')

#Inspect data
print(df.head())
print(df.isnull().sum())
print(df.dtypes)
