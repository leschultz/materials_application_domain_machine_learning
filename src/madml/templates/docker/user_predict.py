from madml.hosting.docker import dockerhub_model
import numpy as np

container_name = 'replace'
model = dockerhub_model(container_name)

X = np.loadtxt('X.csv', delimiter=',')
y = model.predict(X)

print(y)
