
import numpy as np

from mad.datasets import aggregate, load_data, statistics
from mad.ml import MLKR_KNN, feature_selectors, predict, splitters
from mad.plots import parity, versus
from sklearn import cluster
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



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
    # ML setup
    scale = StandardScaler()
    inner_split = splitters.RepeatedClusterSplit(cluster.OPTICS, 5, 2)
    outer_split = splitters.RepeatedClusterSplit(cluster.OPTICS, 5, 2)
    selector = feature_selectors.no_selection()

    # Do KNN
    model = BaggingRegressor( MLKR_KNN.MLKR_KNN())
    grid = {}
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    knn = GridSearchCV(pipe, grid, cv=inner_split)

    # Make pipeline
    pipes = [knn]

    # Evaluate
    predict.run(X, y, outer_split, pipes, save, seed)  # Perform ML
    aggregate.folds(save)  # Combine split data from directory recursively
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save)  # Make parity plots
    versus.make_plots(save, points=points, sampling=sampling)


if __name__ == '__main__':
    main()
