from madml.hosting.docker import predict
import numpy as np

container_name = 'replace'

X = np.loadtxt('X.csv', delimiter=',')

y = predict(X, container_name)
print(y)
