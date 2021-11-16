from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from mad.datasets import aggregate, statistics
from mad.plots import versus
from mad.plots import parity

from mad.datasets import load_data
from mad.ml import  predict, feature_selectors

import numpy as np
import warnings
warnings.filterwarnings('ignore')


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

    grouping = data['class_name']

    # ML setup
    scale = feature_selectors.sequential_vif_selection(n_features=22)
    outer_split = KFold(n_splits=5)
    inner_split = KFold(n_splits=5)
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
    lasso = GridSearchCV(pipe, grid, cv=inner_split)

    # Make pipeline
    pipes = [lasso]

    # Evaluates
    predict.run(X, y, outer_split, pipes, save, seed, groups=grouping)
    aggregate.folds(save)  # Combine split data from directory recursively
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save)  # Make parity plots
    versus.make_plots(save, points, sampling)


if __name__ == '__main__':
    main()
