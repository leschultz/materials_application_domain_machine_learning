from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cluster

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from matplotlib import pyplot as pl
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np
import random
import json
import os

from functions import parallel


def parity(mets, y, y_pred, y_pred_sem, name, units, save):
    '''
    Make a paroody plot.
    '''

    rmse_sig = mets['result_mean'][r'$RMSE/\sigma$']
    rmse_sig_sem = mets['result_sem'][r'$RMSE/\sigma$']

    rmse = mets['result_mean'][r'$RMSE$']
    rmse_sem = mets['result_sem'][r'$RMSE$']

    mae = mets['result_mean'][r'$MAE$']
    mae_sem = mets['result_sem'][r'$MAE$']

    r2 = mets['result_mean'][r'$R^{2}$']
    r2_sem = mets['result_sem'][r'$R^{2}$']

    # Parody plot
    label = r'$RMSE/\sigma=$'
    label += r'{:.2} $\pm$ {:.1}'.format(rmse_sig, rmse_sig_sem)
    label += '\n'
    label += r'$RMSE=$'
    label += r'{:.2} $\pm$ {:.1}'.format(rmse, rmse_sem)
    label += '\n'
    label += r'$MAE=$'
    label += r'{:.2} $\pm$ {:.1}'.format(mae, mae_sem)
    label += '\n'
    label += r'$R^{2}=$'
    label += r'{:.2} $\pm$ {:.1}'.format(r2, r2_sem)

    fig, ax = pl.subplots()
    ax.errorbar(
                y,
                y_pred,
                yerr=y_pred_sem,
                linestyle='none',
                marker='.',
                zorder=0,
                label='Data'
                )

    ax.text(
            0.55,
            0.05,
            label,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black')
            )

    limits = []
    limits.append(min(min(y), min(y_pred))-0.25)
    limits.append(max(max(y), max(y_pred))+0.25)

    # Line of best fit
    ax.plot(
            limits,
            limits,
            label=r'$45^{\circ}$ Line',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(loc='upper left')

    ax.set_ylabel('Predicted {} {}'.format(name, units))
    ax.set_xlabel('Actual {} {}'.format(name, units))

    fig.tight_layout()
    fig.savefig(os.path.join(save, '{}_parity.png'.format(name)))
    pl.close(fig)

    data = {}
    data['y_pred'] = list(y_pred)
    data['y_pred_sem'] = list(y_pred_sem)
    data['y'] = list(y)
    data['metrics'] = mets.to_dict()

    jsonfile = os.path.join(save, 'parity.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def eval_metrics(y, y_pred):
    '''
    Evaluate standard prediction metrics.
    '''

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sig = rmse/np.std(y)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    results = {
               'result': [rmse, rmse_sig, mae, r2],
               'metric': [r'$RMSE$', r'$RMSE/\sigma$', r'$MAE$', r'$R^{2}$']
               }

    return results


def distance_link(X_train, X_test, dist_type):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.

    ouputs:
        dists = A dictionary of distances.
    '''

    # Get the inverse of the covariance matrix from training
    dists = {}
    if dist_type == 'mahalanobis':
        vi = np.linalg.inv(np.cov(X_train.T))
        dist = cdist(X_train, X_test, dist_type, VI=vi)
    else:
        dist = cdist(X_train, X_test, dist_type)

    dists[dist_type+'_mean'] = np.mean(dist, axis=0)
    dists[dist_type+'_max'] = np.max(dist, axis=0)
    dists[dist_type+'_min'] = np.min(dist, axis=0)

    return dists


def distance(X_train, X_test):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    dists = {}
    for i in ['mahalanobis', 'euclidean', 'cityblock']:
        dists.update(distance_link(X_train, X_test, i))

    return dists


def stats(df, cols):
    '''
    Get the statistic of a dataframe.
    '''

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)

    return df


def inner(indx, X, y, gpr, rf):
    '''
    The inner loop from nested cross validation.
    '''

    df = {}

    tr_indx, te_indx = indx
    X_train, X_test = X[tr_indx], X[te_indx]
    y_train, y_test = y[tr_indx], y[te_indx]

    # Calculate distances from test set cases to traning set
    for key, value in distance(X_train, X_test).items():
        if key in df:
            df[key] += list(value)
        else:
            df[key] = list(value)

    gpr.fit(X_train, y_train)
    gpr_best = gpr.best_estimator_

    rf.fit(X_train, y_train)
    rf_best = rf.best_estimator_

    # Get GPR predictions
    y_test_gpr_pred, gpr_std = gpr_best.predict(X_test, return_std=True)

    # Get RF predictions
    y_test_rf_pred = rf_best.predict(X_test)
    rf_scale = rf_best.named_steps['scaler']
    rf_X = rf_best.named_steps['scaler'].transform(X_test)
    rf_estimators = rf_best.named_steps['model'].estimators_
    rf_std = [i.predict(rf_X) for i in rf_estimators]
    rf_std = np.std(rf_std, axis=0)

    gpr_mets = eval_metrics(y_test, y_test_gpr_pred)
    rf_mets = eval_metrics(y_test, y_test_rf_pred)

    df['actual'] = y_test
    df['gpr_pred'] = y_test_gpr_pred
    df['gpr_std'] = gpr_std
    df['rf_pred'] = y_test_rf_pred
    df['rf_std'] = rf_std
    df['index'] = te_indx

    return df, gpr_mets, rf_mets


def outer(split, gpr, rf, X, y, save):
    '''
    Save the true values, predicted values, distances, and model error.
    '''

    # Gather split data in parallel
    data = parallel(inner, list(split.split(X)), X=X, y=y, gpr=gpr, rf=rf)

    # Format data correctly
    df = [pd.DataFrame(i[0]) for i in data]
    gpr_mets = [pd.DataFrame(i[1]) for i in data]
    rf_mets = [pd.DataFrame(i[2]) for i in data]

    # Combine frames
    df = pd.concat(df)
    gpr_mets = pd.concat(gpr_mets)
    rf_mets = pd.concat(rf_mets)

    # Get statistics
    dfstats = stats(df, 'index')
    gpr_metsstats = stats(gpr_mets, 'metric')
    rf_metsstats = stats(rf_mets, 'metric')

    # Gather parity plots
    parity(
           gpr_metsstats,
           dfstats.actual_mean,
           dfstats.gpr_pred_mean,
           dfstats.gpr_pred_sem,
           'GPR',
           '',
           save
           )

    parity(
           rf_metsstats,
           dfstats.actual_mean,
           dfstats.rf_pred_mean,
           dfstats.rf_pred_sem,
           'RF',
           '',
           save
           )

    # Convert series to dataframes
    dfstats = dfstats.reset_index()
    gpr_metsstats = gpr_metsstats.reset_index()
    rf_metsstats = rf_metsstats.reset_index()

    # Save data
    df.to_csv(
              os.path.join(save, 'data.csv'),
              index=False
              )
    dfstats.to_csv(
                   os.path.join(save, 'data_stats.csv'),
                   index=False
                   )
    gpr_mets.to_csv(
                    os.path.join(save, 'gpr_metrics.csv'),
                    index=False
                    )
    gpr_metsstats.to_csv(
                         os.path.join(save, 'gpr_metrics_stats.csv'),
                         index=False
                         )
    rf_mets.to_csv(
                   os.path.join(save, 'rf_metrics.csv'),
                   index=False
                   )
    rf_metsstats.to_csv(
                        os.path.join(save, 'rf_metrics_stats.csv'),
                        index=False
                        )


class splitters:
    '''
    A class used to handle splitter types.
    '''

    def repkf(*argv, **kargv):
        '''
        Repeated K-fold cross validation.
        '''

        return RepeatedKFold(*argv, **kargv)

    def repcf(*argv, **kargv):
        '''
        Custom cluster splitter by fraction.
        '''

        return clust_split(*argv, **kargv)


class clust_split:
    '''
    Custom slitting class which pre-clusters data and then splits on a fraction.
    '''

    def __init__(self, clust, reps, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
            reps = The number of times to apply splitting.
        '''

        self.clust = clust(*args, **kwargs)
        self.reps = reps

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.reps

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order, 
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
            
        outputs:
            A generator for train and test splits.
        '''

        self.clust.fit(X)

        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_

        order = list(set(self.clust.labels_))
        n_clusts = len(order)
        split_num = X.shape[0]//n_clusts

        for rep in range(self.reps):

            # Shuffling
            random.shuffle(order)  # Cluster order
            df = df.sample(frac=1)  # Sample order

            test = []
            train = []
            for i in order:

                data = df.loc[df['cluster'] == i]
                for j in data.index:
                    if len(test) < split_num:
                        test.append(j)
                    else:
                        train.append(j)

            yield train, test


def ml(loc, target, drop, save):
    '''
    Define the machine learning workflow with nested cross validation
    for gaussian process regression and random forest.
    '''

    # Output directory creation
    os.makedirs(save, exist_ok=True)

    # Data handling
    if 'xlsx' in loc:
        df = pd.read_excel(loc)
    else:
        df = pd.read_csv(loc)

    df.drop(drop, axis=1, inplace=True)

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    # ML setup
    scale = StandardScaler()
    split = splitters.repcf(cluster.KMeans, 3, n_clusters=10, n_jobs=1)

    # Gaussian process regression
    kernel = RBF()
    model = GaussianProcessRegressor()
    grid = {}
    grid['model__alpha'] = [1e-1]  # np.logspace(-2, 2, 5)
    grid['model__kernel'] = [RBF(i) for i in np.logspace(-2, 2, 5)]
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    gpr = GridSearchCV(pipe, grid, cv=split)

    # Random forest regression
    model = RandomForestRegressor()
    grid = {}
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    rf = GridSearchCV(pipe, grid, cv=split)

    # Nested CV
    outer(split, gpr, rf, X, y, save)
