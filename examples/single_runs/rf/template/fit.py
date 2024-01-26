from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from madml.models import dissimilarity, calibration, domain, combine
from madml.splitters import BootstrappedLeaveClusterOut
from madml.assess import nested_cv
from madml import datasets
import random


def main():

    run_name = 'run'
    data_name = replace_data
    seed = 0

    # Seed for reproducibility
    random.seed(seed)

    # Load data
    data = datasets.load(data_name)
    X = data['data']
    y = data['target']
    g = data['class_name']
    n_repeats = 5

    # ML Distance model
    ds_model = dissimilarity(dis='kde')

    # ML UQ function
    uq_model = calibration(params=[0.0, 1.0])

    # ML
    scale = StandardScaler()
    model = RandomForestRegressor()

    # The grid for grid search
    grid = {}
    grid['model__n_estimators'] = [100]

    # The machine learning pipeline
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('model', model),
                           ])

    # The gridsearch model
    gs_model = GridSearchCV(
                            pipe,
                            grid,
                            cv=((slice(None), slice(None)),),  # No splits
                            )

    # Types of sampling to test
    splits = [('fit', RepeatedKFold(n_repeats=n_repeats))]

    # Boostrap, cluster data, and generate splits
    for i in [2, 3]:

        # Cluster Splits
        top_split = BootstrappedLeaveClusterOut(
                                                AgglomerativeClustering,
                                                n_repeats=n_repeats,
                                                n_clusters=i
                                                )

        splits.append(('agglo_{}'.format(i), top_split))

    # Assess models
    model = combine(gs_model, ds_model, uq_model, splits)
    cv = nested_cv(model, X, y, splitters=splits)
    df = cv.test()

    # Full fit model and write results.
    cv.write_results(run_name)


if __name__ == '__main__':
    main()
