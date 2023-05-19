from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.models.space import distance_model
from mad.models.combine import domain_model
from mad.models.uq import ensemble_model
from mad.ml.assessment import nested_cv
from mad.datasets import load_data

import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):

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
        uq_model = ensemble_model()

        # ML Pipeline
        scale = StandardScaler()
        model = RandomForestRegressor()

        grid = {}
        grid['model__n_estimators'] = [100]
        grid['model__max_features'] = [None]
        grid['model__max_depth'] = [None]
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('model', model)
                               ])
        gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=1))

        # Types of sampling to test
        splits = [('random', RepeatedKFold(n_repeats=1))]

        # Fit models
        model = domain_model(gs_model, ds_model, uq_model, splits)
        cv = nested_cv(X, y, g, model, splits, save=run_name)
        cv.assess()

        shutil.rmtree(run_name)


if __name__ == '__main__':
    unittest.main()
