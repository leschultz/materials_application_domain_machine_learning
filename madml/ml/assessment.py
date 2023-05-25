from madml.utils import parallel
from madml import plots
import pandas as pd
import numpy as np
import copy
import dill
import os


class nested_cv:
    '''
    Class to do nested CV.
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

        '''
        A class to split data into multiple levels.

        inputs:
            X = The original features to be split.
            y = The original target features to be split.
            g = The groups of data to be split.
        '''

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
        splits = self.split()
        self.splits = list(splits)

    def split(self):
        '''
        Generate the splits.
        '''

        # Train, test splits
        for splitter in self.splitters:
            for count, split in enumerate(splitter[1].split(
                                                            self.X,
                                                            self.y,
                                                            self.g
                                                            )):
                train, test = split
                yield (train, test, count, splitter[0])

    def cv(self, split):
        '''
        Fit models and get predictions from one split.
        '''

        train, test, count, name = split  # train/test

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

    def assess(self):
        '''
        Gather assessment data and plot results.
        '''

        # Assess model
        df = parallel(self.cv, self.splits)
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

        # Save model
        self.model.save = os.path.join(self.save, 'model')
        self.model.fit(self.X, self.y, self.g)
        dill.dump(
                  self.model,
                  open(os.path.join(self.model.save, 'model.dill'), 'wb')
                  )
        self.model.save = None

        return df