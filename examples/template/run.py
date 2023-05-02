from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.models.space import distance_model
from mad.models.uq import ensemble_model
from mad.ml.assessment import combine
from mad.datasets import load_data
from mad.ml import splitters

import pandas as pd
import numpy as np

import dill


def main():

    run_name = 'run'

    # Load data
    data = load_data.replace_data
    X = data['data']
    y = data['target']
    g = data['class_name']
    n_repeats = 1

    # ML Distance model
    ds_model = distance_model(dist='kde')

    # ML UQ function
    uq_model = ensemble_model(params=[0.0, 1.0])

    # ML Pipeline
    scale = StandardScaler()
    model = RandomForestRegressor()

    grid = {}
    grid['model__n_estimators'] = [100]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('model', model),
                           ])
    gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=1))

    # Types of sampling to test
    splits = [('calibration', RepeatedKFold(n_repeats=n_repeats))]

    for i in [2, 3]:

        # Cluster Splits
        top_split = splitters.RepeatedClusterSplit(
                                                   AgglomerativeClustering,
                                                   n_repeats=n_repeats,
                                                   n_clusters=i
                                                   )

        splits.append(('agglo_{}'.format(i), top_split))

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
                      splits,  # Inner loop
                      i[1],  # Outer loop
                      save=save,
                      )

        spl.assess()
        spl.save_model()

        # Load and use model
        with open('{}/model/model.dill'.format(save), 'rb') as in_strm:
            model = dill.load(in_strm)

        print('Loaded model predictions')
        print(pd.DataFrame(model.predict(X)))

    spl.aggregate(run_name)


if __name__ == '__main__':
    main()