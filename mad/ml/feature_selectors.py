from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


class no_selection:
    '''
    A class used to skip feature selection.
    '''

    def transform(self, X):
        '''
        Select all columns
        '''

        return X

    def fit(self, X, y=None):
        return self
