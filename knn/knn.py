# K Nearest Neighbors implementation based on YT series by Sentdex
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

csv = str(raw_input("Filename (csv)?: " ))
df = pd.read_csv(csv)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Shuffle data and use 20% to test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: {}".format(accuracy))

def predict_example():
    # Example using Wisconsin breast cancer dataset
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    # Dataset's 'class' column: 2 = benign, 4 = malignant
    example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    example_measures = example_measures.reshape(len(example_measures),-1)

    prediction = clf.predict(example_measures)
    print("Prediction: {}".format(prediction))
