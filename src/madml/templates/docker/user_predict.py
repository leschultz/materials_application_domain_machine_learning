from madml.hosting.docker import predict
from madml import datasets

container_name = 'replace'

data = container_name.split(':')[-1]
data = datasets.load(data)
X = data['data']

y = predict(X, container_name)
print(y)
