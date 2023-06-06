from madml.datasets import load_data
from madml.ml.docker import predict


container_name = 'leschultz/cmg:latest'

data = load_data.diffusion()
X = data['data']

y = predict(X, container_name)
print(y)
