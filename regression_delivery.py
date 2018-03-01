import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
plt.scatter(delivery['data'][:, 0], delivery['target'])
plt.xlabel('Number of cases')
plt.ylabel('Time')

# Plotting y with x2
plt.figure(2)
plt.scatter(delivery['data'][:, 1], delivery['target'])
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
print("Parameter estimates are", lr.coef_)
x = [int(x) for x in input("Input the number cases followed by the distance. (Separated by a space) ").split(" ")]
x = np.array([x])
print("Predicted value is", lr.predict(x)[0])

mse = mean_squared_error(y_test, lr.predict(x_test))
print("The mean square error is", mse)
