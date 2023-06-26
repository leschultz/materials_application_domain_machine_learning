import pandas as pd
import dill

df = pd.read_csv('/mnt/test.csv')

with open('model.dill', 'rb') as handle:
    model = dill.load(handle)

df = model.predict(df.values)
df.to_csv('/mnt/prediction.csv', index=False)
