from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml.splitters import BootstrappedLeaveClusterOut
from mad.models.space import distance_model
from mad.models.combine import domain_model
from mad.models.uq import calibration_model
from mad.ml.assessment import nested_cv
from mad.datasets import load_data


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
    calibration_model = calibration_model(params=[0.0, 1.0])

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
    gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=1))

    # Types of sampling to test
    splits = [('calibration', RepeatedKFold(n_repeats=n_repeats))]

    # Boostrap, cluster data, and generate splits
    for i in [2, 3]:

        # Cluster Splits
        top_split = BootstrappedLeaveClusterOut(
                                                AgglomerativeClustering,
                                                n_repeats=n_repeats,
                                                n_clusters=i
                                                )

        splits.append(('agglo_{}'.format(i), top_split))

    # Fit models
    model = domain_model(gs_model, ds_model, calibration_model, splits)
    cv = nested_cv(X, y, g, model, splits, save=run_name)
    cv.assess()


if __name__ == '__main__':
    main()
