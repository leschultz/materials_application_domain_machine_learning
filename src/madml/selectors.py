from kneed import KneeLocator
import numpy as np
import copy
import shap


class ShapFeatureSelector:
    '''
    A feature selector that uses SHapley Additive exPlanations.
    '''

    def __init__(self, model, num_features=None):
        '''
        inputs:
            model = The estimator of interest.
            num_features = The number of features to keep.
        '''

        self.model = copy.deepcopy(model)
        self.num_features = num_features

    def fit(self, X, y):
        '''
        inputs:
            X = The features.
            y = The target vaiable.
        '''

        self.model.fit(X, y)

        explainer = shap.Explainer(self.model, X)
        self.scores = explainer(X, check_additivity=False)
        self.scores = np.abs(self.scores.values).mean(axis=0)
        self.scores /= np.sum(self.scores)
        sort = np.argsort(self.scores)[::-1]

        if self.num_features is None:
            knee = KneeLocator(
                               range(len(self.scores)),
                               self.scores[sort],
                               curve='convex',
                               direction='decreasing'
                               )
            self.num_features = knee.knee+1

        self.indx = sort[:self.num_features]

        return self

    def get_scores(self):
        return self.scores

    def transform(self, X):
        '''
        outputs:
            X_selected = The subselected features
        '''

        X_selected = X[:, self.indx]

        return X_selected
