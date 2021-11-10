from mad.functions import parallel, llh, set_llh
from mad.ml import distances

import pandas as pd
import numpy as np
import glob
import os

import warnings
warnings.filterwarnings('ignore')


class builder:
    '''
    Class to use the ingredients of splits to build a model and assessment.
    '''

    def __init__(self, pipe, X, y, d, top_splitter, mid_splitter, save):
        '''
        inputs:
            pipe = The machine learning pipeline.
            X = The features.
            y = The target variable.
            d = The domain for each case.
            top_splitter = The top level domain splitter.
            mid_splitter = The middle level splitter for nested cv.
            splitters = The splitting oject to create 2 layers.
            save = The directory to save splits.
        '''

        self.pipe = pipe
        self.X = X
        self.y = y
        self.d = d
        self.top_splitter = top_splitter
        self.mid_splitter = mid_splitter

        # Output directory creation
        self.save = save

    def assess_domain(self):
        '''
        Asses the model through nested CV with a domain layer.
        '''

        # Renaming conviance
        X, y, d = (self.X, self.y, self.d)
        top = self.top_splitter
        mid = self.mid_splitter
        pipe = self.pipe

        o = np.array(range(X.shape[0]))  # Tracking cases.

        # Setup saving directory.
        save = os.path.join(self.save, 'splits')
        os.makedirs(save, exist_ok=True)

        # Make all of the train, test in domain, and test other domain splits.
        splits = []
        for id_index, od_index in top.split(X, y, d):

            X_id, X_od = X[id_index], X[od_index]
            y_id, y_od = y[id_index], y[od_index]
            d_id, d_od = d[id_index], d[od_index]
            o_id, o_od = o[id_index], o[od_index]

            for i in mid.split(X_id, y_id, d_id):

                tr_index = o_id[i[0]]  # The in domain train.
                te_index = o_id[i[1]]  # The in domain test.
                teod_index = od_index  # The other domain.

                trid_teid_teod = (tr_index, te_index, teod_index)
                splits.append(trid_teid_teod)

        split_counts = range(len(splits))

        # Do nested CV
        parallel(
                 self.nestedcv,
                 list(zip(splits, split_counts)),
                 X=X,
                 y=y,
                 d=d,
                 pipe=pipe,
                 save=save,
                 )


    def nestedcv(
                 self,
                 indexes,
                 X,
                 y,
                 d,
                 pipe,
                 save,
                 ):
        '''
        A class for nesetd cross validation.

        inputs:
            indexes = The in domain test and training indexes.
            X = The feature set.
            y = The target variable.
            d = The class.
            pipe = The machine learning pipe.
            save = The saving directory.

        outputs:
            df = The dataframe for all evaluation.
        '''

        # Split indexes and spit count
        indexes, split_count = indexes
        trid, teid, teod = indexes

        X_id_train, X_id_test, X_od_test = X[trid], X[teid], X[teod]
        y_id_train, y_id_test, y_od_test = y[trid], y[teid], y[teod]
        d_id_train, d_id_test, d_od_test = d[trid], d[teid], d[teod]

        # Calculate distances.
        df_id = distances.distance(X_id_train, X_id_test)
        df_od = distances.distance(X_id_train, X_od_test)

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
        X_od_test_trans = pipe_best_scaler.transform(X_od_test)

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
            y_od_test_pred = pipe_best.predict(X_od_test)

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
            llh_od_test = llh(std_od_test, y_od_test-y_od_test_pred, [a, b])

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
            llh_od_test = llh(std_od_test, y_od_test-y_od_test_pred, [a, b])

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
        df_od['in_domain'] = [False]*X_od_test.shape[0]

        # Grab indexes of tests.
        df_id['index'] = teid
        df_od['index'] = teod

        # Grab the domain of tests.
        df_id['domain'] = d_id_test
        df_od['domain'] = d_od_test

        # Grab the true target variables of test.
        df_id['y'] = y_id_test
        df_od['y'] = y_od_test

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
        df['split_count'] = split_count

        name = 'split_{}.csv'.format(split_count)
        name = os.path.join(
                            save,
                            name
                            )

        df.to_csv(name, index=False)

    def aggregate(self):
        '''
        Gather all data from domain analysis.
        '''

        files = glob.glob(self.save+'/splits/*')

        df = parallel(pd.read_csv, files)
        df = pd.concat(df)

        name = os.path.join(self.save, 'aggregate')
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'data.csv')
        df.to_csv(name, index=False)

        return df
