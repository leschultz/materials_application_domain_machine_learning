from matplotlib import pyplot as pl

import pandas as pd
import json
import os

from mad.functions import parallel


def parity(mets, y, y_pred, y_pred_sem, name, units, save):
    '''
    Make a paroody plot.

    inputs:
        mets = The regression metrics.
        y = The true target value.
        y_pred = The predicted target value.
        y_pred_sem = The standard error of the mean in predicted values.
        name = The name of the target value.
        units = The units of the target value.
        save = The directory to save plot.
    '''

    os.makedirs(save, exist_ok=True)

    rmse_sigma = mets[r'$RMSE/\sigma$_mean']
    rmse_sigma_sem = mets[r'$RMSE/\sigma$_sem']

    rmse = mets[r'$RMSE$_mean']
    rmse_sem = mets[r'$RMSE$_sem']

    mae = mets[r'$MAE$_mean']
    mae_sem = mets[r'$MAE$_sem']

    r2 = mets[r'$R^{2}$_mean']
    r2_sem = mets[r'$R^{2}$_sem']

    label = r'$RMSE/\sigma=$'
    label += r'{:.2} $\pm$ {:.2}'.format(rmse_sigma, rmse_sigma_sem)
    label += '\n'
    label += r'$RMSE=$'
    label += r'{:.2} $\pm$ {:.2}'.format(rmse, rmse_sem)
    label += '\n'
    label += r'$MAE=$'
    label += r'{:.2} $\pm$ {:.2}'.format(mae, mae_sem)
    label += '\n'
    label += r'$R^{2}=$'
    label += r'{:.2} $\pm$ {:.2}'.format(r2, r2_sem)

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
    min_range = min(min(y), min(y_pred))
    max_range = max(max(y), max(y_pred))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)

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
    fig.savefig(os.path.join(save, 'parity.png'.format(name)))
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['y_pred'] = list(y_pred)
    data['y_pred_sem'] = list(y_pred_sem)
    data['y'] = list(y)
    data['metrics'] = mets

    jsonfile = os.path.join(save, 'parity.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def make_plots(save):
    '''
    Define the machine learning workflow with nested cross validation
    for gaussian process regression and random forest.
    '''

    path = os.path.join(save, 'aggregate')
    df = os.path.join(path, 'data_stats.csv')
    mets = os.path.join(path, 'metrics_stats.csv')

    groups = ['scaler', 'model', 'spliter']
    drop_cols = groups+['pipe', 'index']

    df = pd.read_csv(df)
    mets = pd.read_csv(mets)

    for d, m in zip(df.groupby(groups), mets.groupby(groups)):

        name = '_'.join(d[0])
        d = d[1]
        m = m[1]
        m.drop(groups, axis=1, inplace=True)

        m = m.to_dict('records')[0]  # Should have only one entry

        parity(
               m,
               d['y_test_mean'],
               d['y_test_pred_mean'],
               d['y_test_pred_sem'],
               '',
               '',
               os.path.join(path, name)
               )