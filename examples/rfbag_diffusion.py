from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run_rfbag_diffusion'
    points = 100
    uq_coeffs_start = [0.0, 1.1, 0.0]

    # Load data
    data = load_data.diffusion()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']

    # Splitters
    top_split = splitters.BootstrappedLeaveOneGroupOut(2, d)
    mid_split = RepeatedKFold(5, 2)
    bot_split = RepeatedKFold(5, 1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Do LASSO
    model = BaggingRegressor(base_estimator=RandomForestRegressor())
    grid = {}
    grid['model__base_estimator__n_estimators'] = [100]
    grid['model__base_estimator__max_features'] = [None]
    grid['model__base_estimator__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    bagrf = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain.builder(
                            bagrf,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save,
                            seed=seed,
                            uq_coeffs_start=uq_coeffs_start
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save)  # Make parity plots
    calibration.make_plots(save, points, 'std', 'pdf')
    calibration.make_plots(save, points, 'stdcal', 'pdf')


if __name__ == '__main__':
    main()