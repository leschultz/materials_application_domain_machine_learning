from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain, build
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration
from mad.functions import poly

import numpy as np
import dill
import os


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run_rf_diffusion_model'
    points = 15
    uq_func = poly
    uq_coeffs_start = [0.1, 1.1, 0.1, 0.1]

    # Load data
    data = load_data.diffusion_train()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']

    # Splitters
    top_split = splitters.BootstrappedLeaveOneGroupOut(n_repeats=2, groups=d)
    mid_split = RepeatedKFold(n_splits=5, n_repeats=2)
    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Random forest regression
    grid = {}
    model = RandomForestRegressor()
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    rf = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain.builder(
                            rf,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save,
                            seed=seed,
                            uq_func=uq_func,
                            uq_coeffs_start=uq_coeffs_start
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save, 'mahalanobis')  # Make parity plots
    calibration.make_plots(save, points, 'std', 'mahalanobis')
    calibration.make_plots(save, points, 'stdcal', 'mahalanobis')

    # Build model from smaller models and save
    model = build.model(save)
    dill.dump(model, open(os.path.join(save, 'aggregate/model.joblib'), 'wb'))


if __name__ == '__main__':
    main()
