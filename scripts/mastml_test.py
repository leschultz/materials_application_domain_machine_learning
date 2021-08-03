from mastml.data_splitters import SklearnDataSplitter
from mastml.preprocessing import SklearnPreprocessor
from mastml.datasets import LocalDatasets
from mastml.hyper_opt import GridSearch
from mastml.models import SklearnModel
import mastml
import os

input_path = '../original_data/test_small.csv'
data_path = '../data'
out_path = '../mastml_jobs'
target = 'Tc'
error_method = 'stdev_weak_learners'
extra_columns = [
                 'group',
                 'ln(Tc)',
                 ]
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

# Run conditions
preprocessor = SklearnPreprocessor(
                                   preprocessor='StandardScaler',
                                   as_frame=True
                                   )

model = SklearnModel(
                     model='RandomForestRegressor',
                     )

splitter = SklearnDataSplitter(
                               splitter='RepeatedKFold',
                               n_repeats=5,
                               n_splits=5,
                               parallel_run=True,
                               )

hyperopt = GridSearch(
                      param_names='n_estimators',
                      param_values=100,
                      scoring='root_mean_squared_error'
                      )

# Run mastml
splitter.evaluate(
                  X=X,
                  y=y,
                  models=[model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=plots,
                  savepath=out_path,
                  X_extra=X_extra,
                  nested_CV=True,
                  error_method=error_method,
                  verbosity=0,
                  )
