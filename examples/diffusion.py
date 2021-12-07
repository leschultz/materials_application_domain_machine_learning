from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn import cluster

from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain
from mad.datasets import load_data, statistics
from mad.plots import versus, kde, parity, calibration

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run'
    points = None
    sampling = None

    # Load data
    data = load_data.diffusion()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']

    # Splitters
    top_split = splitters.BootstrappedLeaveOneGroupOut(5, d)
    mid_split = RepeatedKFold(5, 1)
    bot_split = RepeatedKFold(5, 1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Do LASSO
    model = BaggingRegressor(base_estimator=Lasso())
    grid = {}
    grid['model__base_estimator__alpha'] = np.logspace(-5, 5, 11)
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    lasso = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain.builder(
                            lasso,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save)  # Make parity plots
    calibration.make_plots(save, 100)


if __name__ == '__main__':
    main()
