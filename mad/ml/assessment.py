from mad.utils import parallel
import pandas as pd
import numpy as np


class nested_cv:

    '''
    A class to split data into multiple levels.

    Parameters
    ----------

    X : numpy array
        The original features to be split.

    y : numpy array
        The original target features to be split.

    g : list or numpy array, default = None
        The groups of data to be split.
    '''

    def __init__(
                 self,
                 X,
                 y,
                 g=None,
                 model=None,
                 splitters=None,
                 ):

        self.X = X  # Features
        self.y = y  # Target
        self.splitters = splitters  # Splitter
        self.model = model

        # Grouping
        if g is None:
            self.g = np.array(['no-groups']*self.X.shape[0])
        else:
            self.g = g

        # Generate the splits
        splits = self.split(
                            self.X,
                            self.y,
                            self.g,
                            self.splitters
                            )
        self.splits = list(splits)

    def split(self, X, y, g, splitters):

        # Train, test splits
        for splitter in splitters:
            for count, split in enumerate(splitter[1].split(X, y, g)):
                train, test = split
                yield (train, test, count, splitter[0])

    def assess(self, split):

        train, test, count, name = split  # train/test

        # Fit models
        self.model.fit(self.X[train], self.y[train], self.g[train])
        data_test = self.model.predict(self.X[test])

        data_test['y'] = self.y[test]
        data_test['g'] = self.g[test]
        data_test['sigma_y'] = self.model.ystd
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'
        data_test['type'] = name
        data_test['y/std(y)'] = data_test['y']/self.model.ystd
        data_test['r'] = data_test['y']-data_test['y_pred']
        data_test['z'] = data_test['r']/data_test['y_stdc']
        data_test['index'] = data_test['index'].astype(int)

        return data_test

    def run(self):
        df = parallel(self.assess, self.splits)
        df = pd.concat(df)
        return df
