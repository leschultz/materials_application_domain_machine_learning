from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from mad.models.space import distance_model
from mad.models.uq import ensemble_model
from mad.ml.assessment import combine
from mad.datasets import load_data
from mad.ml import splitters

import pandas as pd

import unittest
import shutil
import dill


class ml_test(unittest.TestCase):

    def test_ml(self):

        run_name = 'run_gpr'

        # Load data
        data = load_data.diffusion()
        df = data['frame']
        X = data['data']
        y = data['target']
        g = data['class_name']

        # The ground truth choice
        ground = 'calibration'

        # ML Distance model
        ds_model = distance_model(dist='gpr')

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

        # Chemical splits
        n_groups = len(set(g))
        if n_groups > 1:
            chem_split = ('chemical', splitters.LeaveOneGroupOut())
            splits.append(chem_split)

        for i in splits:

            # Assess and build model
            save = '{}/{}'.format(run_name, i[0])

            spl = combine(
                          X,
                          y,
                          g,
                          gs_model,
                          uq_model,
                          ds_model,
                          i[1],
                          ground=ground,
                          save=save,
                          )

            spl.assess()
            spl.save_model()

            # Load and use model
            with open('{}/model/model.dill'.format(save), 'rb') as in_strm:
                model = dill.load(in_strm)

            print('Loaded model predictions')
            print(pd.DataFrame(model.predict(X)))

        # Clean directory
        shutil.rmtree(run_name)


if __name__ == '__main__':
    unittest.main()
