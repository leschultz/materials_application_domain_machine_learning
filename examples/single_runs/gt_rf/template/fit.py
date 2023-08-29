from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from madml.ml.splitters import BootstrappedLeaveClusterOut
from madml.models.space import distance_model
from madml.models.combine import domain_model
from madml.models.uq import calibration_model
from madml.ml.assessment import nested_cv
from madml import datasets


def main():

    run_name = 'run'
    data_name = replace_data

    # Load data
    data = datasets.load(data_name)
    X = data['data']
    y = data['target']
    g = data['class_name']
    n_repeats = 1

    # ML Distance model
    ds_model = distance_model(dist='kde')

    # ML UQ function
    uq_model = calibration_model(params=[0.0, 1.0])

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
    model = domain_model(gs_model, ds_model, uq_model, splits, gts, gtb)
    cv = nested_cv(X, y, g, model, splits, save=run_name)
    cv.assess()
    cv.push('leschultz/cmg:{}'.format(data_name))


if __name__ == '__main__':
    main()
