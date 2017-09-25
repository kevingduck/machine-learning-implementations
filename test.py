# Machine learning modules and implementations
# Linear Regression used to predict stock prices

import datetime
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle
import quandl
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import time
import quandl_api_key

style.use('ggplot') #'fivethirtyeight'

quandl.ApiConfig.api_key = quandl_api_key.get_key()
ticker = raw_input('Enter ticker symbol: ').upper()
df = quandl.get('WIKI/{}'.format(ticker))

df = df[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

forecast_col = 'Adj. Close'

# Instead of getting rid of data missing stuff, just turn them into outliers
df.fillna(-99999, inplace=True)

# Extend forecast out by 1% more of the total data
forecast_out = int(math.ceil(0.05*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

# Drop everything but the label colum to get all features
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X) # Normalize data by scaling
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

# Drop missing data
df.dropna(inplace=True)

# Create label column
y = np.array(df['label'])

# Shuffle up data and use 20% as training data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_jobs is optional; sets number of processes you run at once; -1 uses as many as possible
clf = LinearRegression(n_jobs=-1)

# Easy to switch to another classifier:
# clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)

clf.fit(X_train, y_train)

# Optional: Use pickling instead of re-running classifier every time.
# with open ('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
#
# pickle_in = open('linearregression.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
# print("Forecast_out: {} days".format(forecast_out))

# Create set of forecast data
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Set up forecast column to be populated
df['forecast'] = np.nan

# Hack together date value from timestamp
last_date = df.iloc[-1].name
last_unix = last_date.to_datetime()
last_unix = time.mktime(last_unix.timetuple())
one_day = 86400
next_unix = last_unix + one_day


# Run through forecast_set and make a date
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.title('{} Stock Price + {} days ({}%)'.format(ticker, forecast_out, (np.rint(accuracy*100))))
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
