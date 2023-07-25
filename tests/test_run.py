from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from madml.models.space import distance_model
from madml.models.combine import domain_model
from madml.models.uq import calibration_model
from madml.ml.assessment import nested_cv
from madml import datasets

import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test a quick run.
        '''

        run_name = 'run'

        # Load data
        data = datasets.load('diffusion')
        df = data['frame']
        X = data['data']
        y = data['target']
        g = data['class_name']

        # ML Distance model
        ds_model = distance_model(dist='kde')

        # ML UQ function
        uq_model = calibration_model(params=[0.0, 0.1])

        # ML
        scale = StandardScaler()
        model = RandomForestRegressor()

        # The grid to do grid search
        grid = {}
        grid['model__n_estimators'] = [100]
        grid['model__max_features'] = [None]
        grid['model__max_depth'] = [None]

        # The ML Pipeline
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('model', model)
                               ])

        # The gridsearch model
        gs_model = GridSearchCV(
                                pipe,
                                grid,
                                cv=((slice(None), slice(None)),),  # No splits
                                )

        # Types of sampling to test
        splits = [('calibration', RepeatedKFold(n_repeats=1))]

        # Fit models
        model = domain_model(gs_model, ds_model, uq_model, splits)
        cv = nested_cv(X, y, g, model, splits, save=run_name)
        cv.assess()

        # Clean up directory
        shutil.rmtree(run_name)


if __name__ == '__main__':
    unittest.main()
