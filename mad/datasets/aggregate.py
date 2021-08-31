from pathlib import Path

import pandas as pd
import os

from mad.functions import parallel


def folds(save, low_flag=None):
    '''
    Save aggregate data.

    inputs:
        save = The directory to save and where split data are.
    '''

    path = os.path.join(save, 'splits')
    paths = list(Path(save).rglob('split_*.csv'))

    # Load
    df = parallel(pd.read_csv, paths)
    df = pd.concat(df)

    # Save data
    save = os.path.join(save, 'aggregate')
    os.makedirs(save, exist_ok=True)
    df.to_csv(
              os.path.join(save, 'data.csv'),
              index=False
              )
