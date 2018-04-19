import numpy as np
import neurolab as nl
import pylab as pl

x = list()
y = list()
with open('../datasets/DeliveryTimes.txt') as data:

    for lines in data.readlines()[1:]:
        fields = lines.split()
        y_data, x_data = [float(fields[0])], [float(i) for i in fields[1:]]
        x.append(x_data)
        y.append(y_data)

x = np.array(x)
y = np.array(y)

feat1_constr = [min(x[:,0]), max(x[:,0])]
feat2_constr = [min(x[:,1]), max(x[:,1])]

net = nl.net.newp([feat1_constr, feat2_constr], 1)
error = net.train(x, y, epochs=100, show=10, lr=0.001)

pl.plot(error)
pl.xlabel('epoch number')
pl.ylabel('train error')
pl.grid()
pl.show()


