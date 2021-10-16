import statsmodels.api as sm
import pandas as pd
import numpy as np
import random


class NoSplit:
    '''
    Class to not split data
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

        indx = range(X.shape[0])

        yield indx, indx

class BootstrappedLeaveOneGroupOut:
    '''
    Custom splitting class which with every iteration of n_repeats it will bootstrap the dataset with replacement and leave every group out once with a given class column.
    '''

    def __init__(self, n_repeats, *args, **kwargs):
        '''
        inputs:
            n_repeats = The number of times to apply splitting.
        '''
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups):
        '''
        A method to return the O(N) number of splits.
        '''
        return self.n_repeats * len(set(groups))

    def split(self, X, y=None, groups=None):
        '''
        For every iteration, leave every group out as the testing set one time. 
        '''
        random_state = 0
        df = pd.DataFrame(X)
        grouping_df =  pd.DataFrame(grouping, columns=['group'])
        unique_groups = list(set(groups))
        for rep in range(self.n_repeats):
            bootstrapped_grouping = grouping_df.copy().sample(frac=1, replace=True, random_state=random_state)
            for unique_group in unique_groups:
                if len( bootstrapped_grouping[bootstrapped_grouping['group'] == unique_group]) > 0 :
                    test = bootstrapped_grouping[bootstrapped_grouping['group'] == unique_group].index.tolist()
                    train = bootstrapped_grouping[bootstrapped_grouping['group'] != unique_group].index.tolist()
                    yield train,test
            random_state += 1

    


class RepeatedClusterSplit:
    '''
    Custom splitting class which pre-clusters data and then splits
    to folds.
    '''

    def __init__(self, clust, n_splits, n_repeats, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
            n_splits = The number of splits to apply.
            n_reps = The number of times to apply splitting.
        '''

        self.clust = clust(*args, **kwargs)

        # Make sure it runs in serial
        if hasattr(self.clust, 'n_jobs'):
            self.clust.n_jobs = 1

        self.n_splits = n_splits
        self.n_repeats = n_repeats

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

        self.clust.fit(X)  # Do clustering

        # Get splits based on cluster labels
        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_
        cluster_order = list(set(self.clust.labels_))
        random.shuffle(cluster_order)
        df = df.sample(frac=1)

        # Randomize cluster order
        df = [df.loc[df['cluster'] == i] for i in cluster_order]
        df = pd.concat(df)

        s = np.array_split(df, self.n_splits)  # Split
        range_splits = range(self.n_splits)

        # Do for requested repeats
        for rep in range(self.n_repeats):

            s = [i.sample(frac=1) for i in s]  # Shuffle
            for i in range_splits:

                te = s[i]  # Test
                tr = pd.concat(s[:i]+s[i+1:])  # Train

                # Get the indexes
                train = tr.index.tolist()
                test = te.index.tolist()

                yield train, test


class PDFSplit:
    '''
    Custom splitting class which groups data on a multivariate probability
    distribution function and then splits to folds. Folds should have
    least probable cases.
    '''

    def __init__(self, n_splits, *args, **kwargs):
        '''
        inputs:
            n_splits = The number of splits to apply.
        '''

        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        # Do group based on pdf
        col_types = 'c'*X.shape[-1]  # Assume continuous features
        model = sm.nonparametric.KDEMultivariate(
                                                 X,
                                                 var_type=col_types
                                                 )
        dist = model.pdf(X)

        # Correct return of data
        if isinstance(dist, np.float64):
            dist = [dist]

        df = {'dist': dist, 'index': list(range(X.shape[0]))}
        df = pd.DataFrame(df)
        df.sort_values(by='dist', inplace=True)

        df = np.array_split(df, self.n_splits)
        range_splits = range(self.n_splits)

        for i in range_splits:

            te = df[i]  # Test
            tr = pd.concat(df[:i]+df[i+1:])  # Train

            # Get the indexes
            train = tr.index.tolist()
            test = te.index.tolist()

            yield train, test
