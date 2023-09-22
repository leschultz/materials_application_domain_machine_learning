from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import pandas as pd
import numpy as np

import itertools
import random


class NoSplit:
    '''
    Class to not split data.
    '''

    def get_n_splits(self):
        '''
        Should only have one split.
        '''

        return 1

    def split(self, X, y=None, groups=None):
        '''
        The training and testing indexes are the same.
        '''

        indx = np.array(range(X.shape[0]))

        yield indx, indx


class BootstrappedLeaveOneGroupOut:
    '''
    Custom splitting class which with every iteration of n_repeats it will
    bootstrap the dataset with replacement and leave every group out once
    with a given class column.
    '''

    def __init__(self, n_repeats, groups, *args, **kwargs):
        '''
        inputs:
            n_repeats = The number of times to apply splitting.
            groups =  np.array of group classes for the dataset.
        '''

        self.groups = groups
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the O(N) number of splits.
        '''

        self.groups = groups
        self.n_splits = self.n_repeats*len(set(groups))

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        For every iteration, bootstrap the original dataset, and leave
        every group out as the testing set one time.
        '''

        indx = np.arange(X.shape[0])
        spltr = LeaveOneGroupOut()
        for rep in range(self.n_repeats):

            indx_sample = resample(indx)
            X_sample = X[indx_sample, :]
            y_sample = y[indx_sample]
            g_sample = groups[indx_sample]
            for train, test in spltr.split(X_sample, y_sample, g_sample):
                yield indx_sample[train], indx_sample[test]


class BootstrappedLeaveClusterOut:
    '''
    Custom splitting class which pre-clusters data and then splits
    to folds.
    '''

    def __init__(self, clust, n_repeats, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
            n_repeats = The number of times to apply splitting.
        '''

        self.clust = clust(*args, **kwargs)

        # Make sure it runs in serial
        if hasattr(self.clust, 'n_jobs'):
            self.clust.n_jobs = 1

        self.n_repeats = n_repeats
        self.scaler = StandardScaler()

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits*self.n_repeats

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        indx = np.arange(X.shape[0])
        X = self.scaler.fit_transform(X)
        self.clust.fit(X)  # Do clustering

        # Get splits based on cluster labels
        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_
        cluster_order = list(set(self.clust.labels_))
        random.shuffle(cluster_order)
        df = df.sample(frac=1)

        # Randomize cluster order
        df = [df.loc[df['cluster'] == i].index.tolist() for i in cluster_order]

        self.n_splits = len(cluster_order)
        range_splits = range(self.n_splits)

        # Do for requested repeats
        for rep in range(self.n_repeats):

            sub = [resample(i) for i in df]  # Shuffle
            for i in range_splits:

                test = sub[i]  # Test
                train = sub[:i]+sub[i+1:]  # Train
                train = list(itertools.chain.from_iterable(train))

                train = np.array(train)
                test = np.array(test)

                yield train, test


class LeaveTargetQuantileOut:
    '''
    Group the target variable into a number of quantiles to leave out.
    '''

    def __init__(self, n_splits, *args, **kwargs):
        '''
        inputs:
            n_repeats = The number of times to apply splitting.
            groups =  np.array of group classes for the dataset.
        '''

        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        Grab the target varaible, bin into self.n_splits, and then leave
        one of the bins out while training on the rest.
        '''

        indx = np.arange(X.shape[0])

        df = pd.DataFrame()
        df['indx'] = indx
        df['y'] = y
        df['bin'] = pd.qcut(
                            df['y'].rank(method='first'),
                            self.n_splits
                            )

        for group, values in df.groupby('bin'):

            test = values['indx'].values
            train = [i for i in df['indx'].values if i not in test]
            train = np.array(train)

            yield train, test
