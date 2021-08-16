from matplotlib import pyplot as pl
from itertools import combinations

import seaborn as sns
import pandas as pd
import os

from mad.functions import parallel


def plot(cols, df, save):
    cols = list(cols)
    data = df[cols]

    fig = sns.pairplot(data, kind="kde")
    fig.tight_layout()
    fig.savefig(os.path.join(save, '_'.join(cols)))


def main():
    df = '../original_data/Supercon_data_features_selected.xlsx'
    save = '../analysis/pairplots'
    drop_cols = [
                 'name',
                 'group',
                 'ln(Tc)',
                 ]

    # Output directory creation
    os.makedirs(save, exist_ok=True)

    # Data handling
    df = pd.read_excel(df)
    df.drop(drop_cols, axis=1, inplace=True)

    parallel(plot, list(combinations(df.columns, 2)), df=df, save=save)


if __name__ == '__main__':
    main()
