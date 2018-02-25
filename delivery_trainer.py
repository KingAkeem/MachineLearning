import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class MyModel:

    def __init__(self, b0, b1, b2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

    def predict(self, x1, x2):
        return self.b0+self.b1*x2+self.b2*x2

    def mean_squared_error(self, x_values, y_values):
        s = 0
        for x, y in zip(x_values, y_values):
            s += math.pow(y-self.predict(x[0], x[1]), 2)

        return math.sqrt(s/len(y_values)-3)


dataset = open('./datasets/DeliveryTimes.txt', 'r+')
it = iter(dataset)
delivery = dict()
# Getting column headers for names
column_headers = next(it).split()
delivery['target_name'] = column_headers[0]
delivery['feature_name'] = np.array(column_headers[1:])

# Targets contains all of the y values and data contains the x1 and x2 values
targets = list()
data = list()
for line in it:
    time, cases, dist = (float(x) for x in line.split())
    targets.append(time)
    data.append(np.array([cases, dist]))

# Converting simple list to
delivery['target'] = np.array(targets)
delivery['data'] = np.array(data)

# Plotting y with x1
plt.figure(1)
plt.scatter(delivery['data'][:,0], delivery['target'])
plt.xlabel('Number of cases')
plt.ylabel('Time')

# Plotting y with x2
plt.figure(2)
plt.scatter(delivery['data'][:,1], delivery['target'])
plt.xlabel('Distance')
plt.ylabel('Time')
plt.show()

# Splitting data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(
    delivery['data'],
    delivery['target'],
    random_state=0
)

# Fit Regression model using scikit learn
lr = LinearRegression().fit(x_train, y_train)

y_pred = lr.predict(x_test)
print("ScikitLean model MSE =", mean_squared_error(y_test, y_pred))

# Calculating estimates manually and then creating linear regression equation
# using estimated parameters
beta_one = 0
x1_bar, x2_bar = np.mean(data, axis=0)
y_bar = np.mean(targets)

b1_divisor = 0
b1_dividend = 0

b2_divisor = 0
b2_dividend = 0

for x, y in zip(x_train, y_train):
    b1_divisor += (x[0] - x1_bar)*(y-y_bar)
    b1_dividend += math.pow((x[0] - x1_bar), 2)

    b2_divisor += (x[1] - x2_bar)*(y-y_bar)
    b2_dividend += math.pow((x[1] - x2_bar), 2)

beta_one = b1_divisor/b1_dividend
beta_two = b2_divisor/b2_dividend
beta_zero = y_bar-beta_one*x1_bar
model = MyModel(beta_zero, beta_one, beta_two)
print("My model MSE = ", model.mean_squared_error(x_test, y_test))
