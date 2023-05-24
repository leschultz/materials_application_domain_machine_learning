from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.models.space import distance_model
from mad.models.combine import domain_model
from mad.models.uq import calibration_model
from mad.ml.assessment import nested_cv
from mad.datasets import load_data

import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test a quick run.
        '''

        run_name = 'run'

        # Load data
        data = load_data.diffusion()
        df = data['frame']
        X = data['data']
        y = data['target']
        g = data['class_name']

        # ML Distance model
        ds_model = distance_model(dist='kde')

        # ML UQ function
        calibration_model = calibration_model()

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
        gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=1))

        # Types of sampling to test
        splits = [('random', RepeatedKFold(n_repeats=1))]

        # Fit models
        model = domain_model(gs_model, ds_model, calibration_model, splits)
        cv = nested_cv(X, y, g, model, splits, save=run_name)
        cv.assess()

        # Clean up directory
        shutil.rmtree(run_name)


if __name__ == '__main__':
    unittest.main()
