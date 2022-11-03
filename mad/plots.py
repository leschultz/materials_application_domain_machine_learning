from matplotlib import pyplot as pl

import json
import os


def parity(
           mets,
           y,
           y_pred,
           y_pred_sem=None,
           name='',
           units='',
           save='.'
           ):
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

    mets = mets.to_dict(orient='records')[0]

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

    if y_pred_sem is not None:
        ax.errorbar(
                    y,
                    y_pred,
                    yerr=y_pred_sem,
                    linestyle='none',
                    marker='.',
                    markerfacecolor='None',
                    zorder=0,
                    )

    ax.scatter(
               y,
               y_pred,
               marker='.',
               zorder=1,
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
            label=r'$y=\hat{y}$',
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
    fig.savefig(os.path.join(save, 'parity.png'))
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['y_pred'] = list(y_pred)
    data['y'] = list(y)
    data['metrics'] = mets

    if y_pred_sem is not None:
        data['y_pred_sem'] = list(y_pred_sem)

    jsonfile = os.path.join(save, 'parity.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)
