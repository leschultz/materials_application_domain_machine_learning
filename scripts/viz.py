from mastml.datasets import LocalDatasets
from matplotlib import pyplot as pl
import seaborn as sns
import os

input_path = '../original_data/diffusion.xlsx'
data_path = '../data'
out_path = '../mastml_jobs'
target = 'E_regression'
error_method = 'stdev_weak_learners'
extra_columns = ['Material compositions 1', 'Material compositions 2']
plots = [
         'Error',
         'Scatter',
         'Histogram'
         ]
metrics = [
           'r2_score',
           'mean_absolute_error',
           'root_mean_squared_error',
           'rmse_over_stdev'
           ]

# Load data
d = LocalDatasets(
                  file_path=input_path,
                  target=target,
                  extra_columns=extra_columns,
                  testdata_columns=None,
                  as_frame=True
                  )

# Load the data with the load_data() method
data_dict = d.load_data()

# Let's assign each data object to its respective name
X = data_dict['X']
y = data_dict['y']
X_extra = data_dict['X_extra']

cols = X.columns

X = X[cols[:3]]

print(X)
sns.pairplot(X, kind="kde")

pl.show()
