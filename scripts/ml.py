from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import sys

from matplotlib import pyplot as pl
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np
import json
import os


def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.
    '''

    pool = Pool(os.cpu_count())
    part_func = partial(func, *args, **kwargs)

    data = list(tqdm(pool.imap(part_func, x), total=len(x), file=sys.stdout))
    pool.close()
    pool.join()

    return data


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


def distance(X_train, X_test):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    dists = {}

    # Get the inverse of the covariance matrix from training
    vi = np.linalg.inv(np.cov(X_train.T))
    mu = np.mean(X_train, axis=0)  # Mean of columns

    dist_maha = cdist([mu], X_test, 'mahalanobis', VI=vi)
    dist = cdist(X_train, X_test, 'mahalanobis', VI=vi)
    dist_mean = np.mean(dist, axis=0)
    dist_max = np.max(dist, axis=0)
    dist_min = np.min(dist, axis=0)

    dists['mahalanobis'] = dist_maha.ravel()
    dists['mahalanobis_mean'] = dist_mean
    dists['mahalanobis_max'] = dist_max
    dists['mahalanobis_min'] = dist_min

    dist_euclid = cdist([mu], X_test, 'euclidean')
    dist = cdist(X_train, X_test, 'euclidean')
    dist_mean = np.mean(dist, axis=0)
    dist_max = np.max(dist, axis=0)
    dist_min = np.min(dist, axis=0)

    dists['euclidean'] = dist_maha.ravel()
    dists['euclidean_mean'] = dist_mean
    dists['euclidean_max'] = dist_max
    dists['euclidean_min'] = dist_min

    dist_city = cdist([mu], X_test, 'cityblock')
    dist = cdist(X_train, X_test, 'cityblock')
    dist_mean = np.mean(dist, axis=0)
    dist_max = np.max(dist, axis=0)
    dist_min = np.min(dist, axis=0)

    dists['cityblock'] = dist_maha.ravel()
    dists['cityblock_mean'] = dist_mean
    dists['cityblock_max'] = dist_max
    dists['cityblock_min'] = dist_min

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
    The inner loop.
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
    rf_estimators = rf_best.named_steps['model'].estimators_
    rf_std = [i.predict(X_test) for i in rf_estimators]
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

    data = parallel(inner, list(split.split(X)), X=X, y=y, gpr=gpr, rf=rf)

    df = [pd.DataFrame(i[0]) for i in data]
    gpr_mets = [pd.DataFrame(i[1]) for i in data]
    rf_mets = [pd.DataFrame(i[2]) for i in data]

    df = pd.concat(df)
    gpr_mets = pd.concat(gpr_mets)
    rf_mets = pd.concat(rf_mets)

    dfstats = stats(df, 'index')
    gpr_metsstats = stats(gpr_mets, 'metric')
    rf_metsstats = stats(rf_mets, 'metric')

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

    df.to_csv(os.path.join(save, 'data.csv'), index=False)
    dfstats.to_csv(os.path.join(save, 'data_stats.csv'), index=False)
    gpr_mets.to_csv(os.path.join(save, 'gpr_metrics.csv'), index=False)
    gpr_metsstats.to_csv(os.path.join(save, 'gpr_metrics_stats.csv'), index=False)
    rf_mets.to_csv(os.path.join(save, 'rf_metrics.csv'), index=False)
    rf_metsstats.to_csv(os.path.join(save, 'rf_metrics_stats.csv'), index=False)


def main():

    df = '../original_data/diffusion.xlsx'
    save = '../analysis'
    target = 'E_regression'
    drop_cols = [
                 'Material compositions 1',
                 'Material compositions 2'
                 ]

    # Output directory creation
    os.makedirs(save, exist_ok=True)

    # Data handling
    df = pd.read_excel(df)
    df.drop(drop_cols, axis=1, inplace=True)

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    # ML setup
    scale = StandardScaler()
    split = RepeatedKFold(n_splits=5, n_repeats=2)

    # Gaussian process regression
    kernel = RBF()
    model = GaussianProcessRegressor(kernel=kernel)
    grid = {}
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    gpr = GridSearchCV(pipe, grid)

    # Random forest regression
    model = RandomForestRegressor()
    grid = {}
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    rf = GridSearchCV(pipe, grid)

    # Nested CV
    outer(split, gpr, rf, X, y, save)


if __name__ == '__main__':
    main()
