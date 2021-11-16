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


class vif_selection:
    '''
    A class for VIF score for feature selection.
    '''

    def __init__(self, n_features):
        '''
        inputs:
            n_features = number of features to choose
        '''

        self.n_features = n_features

    def transform(self, X):
        '''
        Find the top percent of features, and transform the data.
        '''

        feature_indices = list(range(X.shape[1]))
        assert(self.n_features <= len(feature_indices)), "n_features ({}) needs to be less than or equal to the amount of features in the dataset ({})".format(
            str(self.n_features), str(len(feature_indices)))
        vif = pd.DataFrame()
        vals = []
        for i in feature_indices:
            vals.append(variance_inflation_factor(X, i))
        vif['VIF Factor'] = vals

        vif['features_index'] = feature_indices
        vif_sorted = vif.sort_values('VIF Factor')
        feature_selected = list(vif_sorted['features_index'])[:self.n_features]

        X = X[:, feature_selected]

        return X

    def fit(self, X, y=None):
        return self

class sequential_vif_selection:
    '''
    A class for VIF score for feature selection. Works with all features, then removes them one by one.
    '''

    def __init__(self, n_features):
        '''
        inputs:
            percent = gives the top percent of feature.
        '''
        
        self.n_features = n_features

    def transform(self, X):
        '''
        Find the top n_features of features, and transform the data.
        '''

        #Instantiate DF , first iteration
        
        vif_df = pd.DataFrame()
        feature_indices = list(range(X.shape[1]))
        assert(self.n_features <= len(feature_indices)), "n_features ({}) needs to be less than or equal to the amount of features in the dataset ({})".format(
            str(self.n_features), str(len(feature_indices)))
        vals = [variance_inflation_factor(X, i) for i in feature_indices]
        vif_df['VIF Factor'] = vals
        vif_df['features_index'] = feature_indices
        vif_sorted = vif_df.sort_values('VIF Factor')
        #remove 1 feature until there are self.n_features left
        while len(vals)> self.n_features:
            # remove 1 feature
            features_selected = list(vif_sorted['features_index'])[:len(vals)-1]
            X = X[:, features_selected]
            if features_selected == self.n_features:
                break

            # Preparation for next iteration; re-calculate VIF of every variable 
            feature_indices = list(range(X.shape[1]))
            vals = [variance_inflation_factor(X, i) for i in feature_indices]
            
            vif_df = pd.DataFrame()
            vif_df['VIF Factor'] = vals
            vif_df['features_index'] = feature_indices
            vif_sorted = vif_df.sort_values('VIF Factor')
        return X

    def fit(self, X, y=None):
        return self
