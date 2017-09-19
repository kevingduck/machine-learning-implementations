# Linear Regression from scratch based on YT series by Sentdex
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean

style = 'fivethirtyeight'

# Made up data ...
xs = np.array([2,5,3,4,6,7,4,5,8,6,7,9,8], dtype=np.float64)
ys = np.array([6,5,6,5,7,6,7,6,8,7,9,8,9], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

# Find slope (m) and y-intercept (b)
def get_best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

m,b = get_best_fit_slope_and_intercept(xs,ys)

# create regression line after m,b are calculated
regression_line = [(m*x)+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
