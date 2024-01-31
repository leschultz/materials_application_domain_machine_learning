from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from madml.models import dissimilarity, calibration, combine
from madml.assess import nested_cv
from madml import datasets

import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test a quick run.
        '''

        run_name = 'output'

        # Load data
        data = datasets.load('strength')
        df = data['frame']
        X = data['data']
        y = data['target']

        # ML Distance model
        ds_model = dissimilarity(dis='kde')

        # ML UQ function
        uq_model = calibration(params=[0.0, 0.1])

        # ML
        scale = StandardScaler()
        model = RandomForestRegressor()

        # The grid to do grid search
        grid = {}
        grid['model__n_estimators'] = [100]

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
        splits = [('fit', RepeatedKFold(n_repeats=1))]

        # Assess models
        model = combine(gs_model, ds_model, uq_model, splits)
        cv = nested_cv(model, X, y, splitters=splits)
        df, df_bin, fit_model = cv.test(
                                        save_inner_folds=run_name,
                                        save_outer_folds=run_name,
                                        )

        # Clean up directory
        shutil.rmtree(run_name)


if __name__ == '__main__':
    unittest.main()
