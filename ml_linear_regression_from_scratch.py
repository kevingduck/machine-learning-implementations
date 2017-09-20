# Linear Regression from scratch based on YT series by Sentdex
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
import random

style = 'fivethirtyeight'

# Set up dataset with how many (hm) variables, etc.
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# Made up data ...
xs = np.array([2,5,3,4,6,7,4,5,8,6,7,9,8], dtype=np.float64)
ys = np.array([6,5,6,5,7,6,7,6,8,7,9,8,9], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

# To caluclate r^2 coefficient_of_determination value use formula:
# r^2 = 1 - (SE regression / SE mean(y))

# Calculate SE, or square of Y distances from points
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

# Find slope (m) and y-intercept (b)
def get_best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

m,b = get_best_fit_slope_and_intercept(xs,ys)

# create regression line after m,b are calculated
regression_line = [(m*x)+b for x in xs]

# Calculate r^2
r_squared = coefficient_of_determination(ys, regression_line)
print("r^2 value: {}".format(r_squared))

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
