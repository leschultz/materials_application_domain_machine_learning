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

import dill


def main():

    run_name = 'run_kde'

    # Load data
    data = load_data.friedman()
    df = data['frame']
    X = data['data']
    y = data['target']
    g = data['class_name']

    sub_test = 0.2  # Fraction of each outer split to be randomly sampled

    # ML Distance model
    ds_model = distance_model(dist='kde')

    # ML UQ function
    uq_model = ensemble_model(params=[0.0, 1.0])

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

    for i in [2]:

        # Cluster Splits
        top_split = splitters.RepeatedClusterSplit(
                                                   KMeans,
                                                   n_init='auto',
                                                   n_repeats=1,
                                                   n_clusters=i
                                                   )

        splits.append(('kmeans_{}'.format(i), top_split))

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
                      sub_test=sub_test,
                      save=save,
                      )

        spl.assess()
        spl.save_model()

        # Load and use model
        with open('{}/model/model.dill'.format(save), 'rb') as in_strm:
            model = dill.load(in_strm)

        print('Loaded model predictions')
        print(pd.DataFrame(model.predict(X)))


if __name__ == '__main__':
    main()
