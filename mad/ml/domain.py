from mad.functions import parallel, llh, set_llh
from mad.ml import distances

import pandas as pd
import numpy as np


class builder:
    '''
    Class to use the ingredients of splits to build a model and assessment.
    '''

    def __init__(self, pipe, X, y, d, splitters):
        '''
        inputs:
            pipe = The machine learning pipeline.
            X = The features.
            y = The target variable.
            d = The domain for each case.
            splitters = The splitting oject to create 3 layers.
        '''

        self.pipe = pipe
        self.X = X
        self.y = y
        self.d = d
        self.top_splitter, self.mid_splitter = splitters

    def assess_model(self):
        '''
        Asses the model through nested CV.
        '''

        # Renaming conviance
        X, y, d = (self.X, self.y, self.d)
        top = self.top_splitter
        mid = self.mid_splitter
        pipe = self.pipe

        o = np.array(range(X.shape[0]))  # Tracking cases

        # In domain (ID) and other domain (OD) splits.
        for id_index, od_index in top.split(X, y, d):

            X_id, X_od = X[id_index], X[od_index]
            y_id, y_od = y[id_index], y[od_index]
            d_id, d_od = d[id_index], d[od_index]
            o_id, o_od = o[id_index], o[od_index]

            splits = list(mid.split(X_id, y_id, d_id))
            counts = list(range(len(splits)))
            parallel(
                     self.nestedcv,
                     list(zip(splits, counts)),
                     X_id=X_id,
                     X_od=X_od,
                     y_id=y_id,
                     y_od=y_od,
                     d_id=d_id,
                     d_od=d_od,
                     o_id=o_id,
                     o_od=o_od,
                     pipe=pipe,
                     )

    def nestedcv(
                 self,
                 indexes,
                 X_id,
                 X_od,
                 y_id,
                 y_od,
                 d_id,
                 d_od,
                 o_id,
                 o_od,
                 pipe
                 ):
        '''
        A class for nesetd cross validation.

        inputs:
            indexes = The in domain test and training indexes.
            X_id = The in domain feature set.
            X_od = The other domain feature set.
            y_id = The in domain target variable.
            y_od = The other domain target variable.
            d_id = The in domain class.
            d_od = The other domain class.
            o_id = The in domain indexes.
            o_od = The other domain indexes.
            pipe = The machine learning pipe.

        outputs:
            df = The dataframe for all evaluation.
        '''

        # Training and testing splits.
        indexes, count = indexes
        tr_index, te_index = indexes

        X_id_train, X_id_test = X_id[tr_index], X_id[te_index]
        y_id_train, y_id_test = y_id[tr_index], y_id[te_index]
        d_id_train, d_id_test = d_id[tr_index], d_id[te_index]
        o_id_train, o_id_test = o_id[tr_index], o_id[te_index]

        # Calculate distances.
        df_id = distances.distance(X_id_train, X_id_test)
        df_od = distances.distance(X_id_train, X_od)

        # Fit the model on training data in domain.
        self.pipe.fit(X_id_train, y_id_train)

        # Grab model critical information for assessment
        pipe_best = pipe.best_estimator_
        pipe_best_scaler = pipe_best.named_steps['scaler']
        pipe_best_select = pipe_best.named_steps['select']
        pipe_best_model = pipe_best.named_steps['model']

        # Grab model specific details
        model_type = pipe_best_model.__class__.__name__
        scaler_type = pipe_best_scaler.__class__.__name__
        split_type = pipe.cv.__class__.__name__

        # Feature transformations
        X_id_train_trans = pipe_best_scaler.transform(X_id_train)
        X_id_test_trans = pipe_best_scaler.transform(X_id_test)
        X_od_test_trans = pipe_best_scaler.transform(X_od)

        # Feature selection
        X_id_train_select = pipe_best_select.transform(X_id_train_trans)
        X_id_test_select = pipe_best_select.transform(X_id_test_trans)
        X_od_test_select = pipe_best_select.transform(X_od_test_trans)

        n_features = X_id_test_select.shape[-1]

        # If model is ensemble regressor
        ensemble_methods = ['RandomForestRegressor', 'BaggingRegressor']
        if model_type in ensemble_methods:
            y_id_train_pred = pipe_best.predict(X_id_train)
            y_id_test_pred = pipe_best.predict(X_id_test)
            y_od_test_pred = pipe_best.predict(X_od)

            # Ensemble predictions with correct feature set
            pipe_estimators = pipe_best_model.estimators_
            std_id_train = []
            std_id_test = []
            std_od_test = []
            for i in pipe_estimators:
                std_id_train.append(i.predict(X_id_train_select))
                std_id_test.append(i.predict(X_id_test_select))
                std_od_test.append(i.predict(X_od_test_select))

            std_id_train = np.std(std_id_train, axis=0)
            std_id_test = np.std(std_id_test, axis=0)
            std_od_test = np.std(std_od_test, axis=0)

            # Calibration.
            a, b = set_llh(std_id_train, y_id_train, y_id_train_pred, [0, 1])
            stdcal_id_test = a*std_id_test+b
            stdcal_od_test = a*std_od_test+b

            # Log likelihoods.
            llh_id_test = llh(std_id_test, y_id_test-y_id_test_pred, [a, b])
            llh_od_test = llh(std_od_test, y_od-y_od_test_pred, [a, b])

            # Grab standard deviations.
            df_id['std'] = std_id_test
            df_od['std'] = std_od_test

            # Grab calibrated standard deviations.
            df_id['stdcal'] = stdcal_id_test
            df_od['stdcal'] = stdcal_od_test

            # Grab the log likelihood values.
            df_id['llh'] = llh_id_test
            df_od['llh'] = llh_od_test

        # If model is gaussian process regressor
        elif model_type == 'GaussianProcessRegressor':
            y_id_train_pred, std_id_train = pipe_best.predict(
                                                              X_id_train,
                                                              return_std=True
                                                              )
            y_id_test_pred, std_id_test = pipe_best.predict(
                                                            X_id_test,
                                                            return_std=True
                                                            )

            y_od_test_pred, std_od_train = pipe_best.predict(
                                                             X_od_test,
                                                             return_std=True
                                                             )

            # Calibration.
            a, b = set_llh(std_id_train, y_id_train, y_id_train_pred, [0, 1])
            stdcal_id_test = a*std_id_test+b
            stdcal_od_test = a*std_od_test+b

            # Log likelihoods.
            llh_id_test = llh(std_id_test, y_id_test-y_id_test_pred, [a, b])
            llh_od_test = llh(std_od_test, y_od-y_od_test_pred, [a, b])

            # Grab standard deviations.
            df_id['std'] = std_id_test
            df_od['std'] = std_od_test

            # Grab calibrated standard deviations.
            df_id['stdcal'] = stdcal_id_test
            df_od['stdcal'] = stdcal_od_test

            # Grab the log likelihood values.
            df_id['llh'] = llh_id_test
            df_od['llh'] = llh_od_test

        # If model does not support standard deviation
        else:
            y_id_test_pred = pipe_best.predict(X_id_test)
            y_od_test_pred = pipe_best.predict(X_od_test)

        # Assign boolean for in domain.
        df_id['in_domain'] = [True]*X_id_test.shape[0]
        df_od['in_domain'] = [False]*X_od.shape[0]

        # Grab indexes of tests.
        df_id['index'] = o_id_test
        df_od['index'] = o_od

        # Grab the domain of tests.
        df_id['domain'] = d_id_test
        df_od['domain'] = d_od

        # Grab the true target variables of test.
        df_id['y'] = y_id_test
        df_od['y'] = y_od

        # Grab the predictions of tests.
        df_id['y_pred'] = y_id_test_pred
        df_od['y_pred'] = y_od_test_pred

        df_id = pd.DataFrame(df_id)
        df_od = pd.DataFrame(df_od)

        df = pd.concat([df_id, df_od])

        # Assign values that should be the same
        df['pipe'] = pipe
        df['model'] = model_type
        df['scaler'] = scaler_type
        df['features'] = n_features
        df['splitter'] = split_type

        return df
