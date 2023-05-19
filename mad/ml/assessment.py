from mad.utils import parallel
from mad import plots
import pandas as pd
import numpy as np
import copy
import dill
import os


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
                 save=None,
                 ):

        self.X = X  # Features
        self.y = y  # Target
        self.splitters = splitters  # Splitter
        self.model = model
        self.save = save

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

        print(count, name)

        # Fit models
        self.model.fit(self.X[train], self.y[train], self.g[train])
        data_test = self.model.predict(self.X[test])

        data_test['y'] = self.y[test]
        data_test['g'] = self.g[test]
        data_test['std(y)'] = self.model.ystd
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'
        data_test['type'] = name
        data_test['index'] = data_test['index'].astype(int)

        # z score
        data_test['r'] = data_test['y']-data_test['y_pred']
        data_test['z'] = data_test['r']/data_test['y_stdc']
        data_test['r/std(y)'] = data_test['r']/data_test['std(y)']

        # Normalized
        data_test['y_stdc/std(y)'] = data_test['y_stdc']/data_test['std(y)']
        data_test['y_pred/std(y)'] = data_test['y_pred']/data_test['std(y)']
        data_test['y/std(y)'] = data_test['y']/data_test['std(y)']

        return data_test

    def run(self):

        '''
        # Save model
        model = copy.deepcopy(self.model)
        model.save = os.path.join(self.save, 'model')
        model.fit(self.X, self.y, self.g)
        dill.dump(model, open(os.path.join(model.save, 'model.dill'), 'wb'))
        del model
        '''

        # Assess model
        df = parallel(self.assess, self.splits)
        df = pd.concat(df)
        df['id'] = abs(df['r/std(y)']) < 1.0

        save = os.path.join(self.save, 'assessment')
        out = plots.generate_plots(
                                   df,
                                   np.std(self.y),
                                   self.model.bins,
                                   save,
                                   )

        df.to_csv(os.path.join(save, 'single.csv'), index=False)

        return df
