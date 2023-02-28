from sklearn.metrics import precision_recall_curve, mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.svm import SVC

from mad.stats.group import stats, group_metrics
from mad.utils import parallel
from mad.ml import splitters
from mad import plots

from sklearn.model_selection import LeaveOneGroupOut
from sklearn import cluster

import pandas as pd
import numpy as np

import copy
import dill
import os


def domain_pred(dist, dist_cut, domain):
    '''
    Predict the domain based on thresholds.
    '''

    do_pred = []
    for i in dist:
        if domain is True:
            if i < dist_cut:
                do_pred.append(True)
            else:
                do_pred.append(False)
        elif domain is False:
            if i >= dist_cut:
                do_pred.append(True)
            else:
                do_pred.append(False)

    return do_pred


def ground_truth(
                 y,
                 y_pred,
                 cut=None,
                 ):

    # Define ground truth
    absres = abs(y-y_pred)

    if cut is None:
        cut = np.percentile(absres, 99)

    do_pred = absres < cut
    do_pred = [True if i == 1 else False for i in do_pred]

    return cut, do_pred


def transforms(gs_model, X):

    for step in list(gs_model.best_estimator_.named_steps)[:-1]:

        step = gs_model.best_estimator_.named_steps[step]
        X = step.transform(X)

    return X


def std_pred(gs_model, X_test):

    estimators = gs_model.best_estimator_
    estimators = estimators.named_steps['model']
    estimators = estimators.estimators_

    std = []
    for i in estimators:
        std.append(i.predict(X_test))

    std = np.std(std, axis=0)

    return std


def cv(gs_model, ds_model, X, y, g, train, cv):
    '''
    Do cross validation.
    '''

    g_cv = []
    y_cv = []
    y_cv_pred = []
    y_cv_std = []
    index_cv = []
    dist_cv = []
    for tr, te in cv.split(
                           X[train],
                           y[train],
                           g[train],
                           ):

        gs_model_cv = clone(gs_model)
        ds_model_cv = copy.deepcopy(ds_model)

        gs_model_cv.fit(X[train][tr], y[train][tr])

        X_trans_tr = transforms(
                                gs_model_cv,
                                X[train][tr],
                                )

        X_trans_te = transforms(
                                gs_model_cv,
                                X[train][te],
                                )

        ds_model_cv.fit(X_trans_tr, y[train][tr])

        std = std_pred(gs_model, X_trans_te)

        y_cv_pred = np.append(
                              y_cv_pred,
                              gs_model_cv.predict(X[train][te])
                              )

        y_cv_std = np.append(
                             y_cv_std,
                             std
                             )
        y_cv = np.append(
                         y_cv,
                         y[train][te]
                         )
        g_cv = np.append(
                         g_cv,
                         g[train][te]
                         )

        index_cv = np.append(index_cv, train[te])
        dist_cv = np.append(
                            dist_cv,
                            ds_model_cv.predict(X_trans_te)
                            )

    data = pd.DataFrame()
    data['g'] = g_cv
    data['y'] = y_cv
    data['y_pred'] = y_cv_pred
    data['y_std'] = y_cv_std
    data['dist'] = dist_cv
    data['index'] = index_cv
    data['split'] = 'cv'

    return data


class build_model:

    def __init__(
                 self,
                 gs_model,
                 ds_model,
                 uq_model,
                 ):

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model

    def fit(self, X, y, g):

        # Get some data statistics
        self.ystd = np.std(y)

        # Build the model
        self.gs_model.fit(X, y)

        X_trans = transforms(
                             self.gs_model,
                             X,
                             )
        self.ds_model.fit(X_trans, y)

        # Do cross validation in nested loop
        data_id = cv(
                     self.gs_model,
                     self.ds_model,
                     X,
                     y,
                     g,
                     np.arange(y.shape[0]),
                     self.gs_model.cv
                     )

        # Fit on hold out data ID
        self.uq_model.fit(
                          data_id['y'],
                          data_id['y_pred'],
                          data_id['y_std']
                          )

        data_id['y_std'] = self.uq_model.predict(data_id['y_std'])

        # Define ground truth
        cut, in_domain = ground_truth(
                                      data_id['y'],
                                      data_id['y_pred'],
                                      )

        self.cut = cut

        data_id['in_domain'] = in_domain

        # OD split
        od_split = splitters.RepeatedClusterSplit(
                                                  KMeans,
                                                  n_init='auto',
                                                  n_repeats=1,
                                                  n_clusters=2
                                                  )
        data_od = cv(
                     self.gs_model,
                     self.ds_model,
                     X,
                     y,
                     g,
                     np.arange(y.shape[0]),
                     od_split
                     )

        data_od['y_std'] = self.uq_model.predict(data_od['y_std'])

        _, in_domain = ground_truth(
                                    data_od['y'],
                                    data_od['y_pred'],
                                    cut=cut,
                                    )

        data_od['in_domain'] = in_domain

        # Combine OD and ID
        data_cv = pd.concat([data_id, data_od])

        self.domain_cut = {'dist': {}, 'y_std': {}}
        for i in [True, False]:

            for j in ['dist', 'y_std']:

                self.domain_cut[j][i] = plots.pr(
                                                 data_cv[j],
                                                 data_cv['in_domain'],
                                                 pos_label=i,
                                                 choice='rel_f1'
                                                 )

                do_pred = domain_pred(
                                      data_cv[j],
                                      self.domain_cut[j][i],
                                      i,
                                      )

                if i is True:
                    data_cv['{}_in_domain_pred'.format(j)] = do_pred
                else:
                    data_cv['{}_out_domain_pred'.format(j)] = do_pred

        self.data_cv = data_cv

        return data_cv

    def predict(self, X):

        X_trans = transforms(
                             self.gs_model,
                             X,
                             )

        # Model predictions
        y_pred = self.gs_model.predict(X)
        y_std = std_pred(self.gs_model, X_trans)
        y_std = self.uq_model.predict(y_std)  # Calibrate hold out
        dist = self.ds_model.predict(X_trans)

        pred = {
                'y_pred': y_pred,
                'y_std': y_std,
                'dist': dist,
                }

        for i in [True, False]:
            for j in ['dist', 'y_std']:
                do_pred = domain_pred(
                                      dist,
                                      self.domain_cut[j][i],
                                      domain=i,
                                      )

                if i is True:
                    pred['{}_in_domain_pred'.format(j)] = do_pred
                else:
                    pred['{}_out_domain_pred'.format(j)] = do_pred

        pred = pd.DataFrame(pred)

        return pred


class combine:

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
                 gs_model=None,
                 uq_model=None,
                 ds_model=None,
                 splitter=RepeatedKFold(),
                 sub_test=0.0,
                 ground='calibration',
                 save='.',
                 ):

        self.X = X  # Features
        self.y = y  # Target
        self.splitter = splitter  # Splitter
        self.sub_test = sub_test
        self.ground = ground

        # Models
        self.gs_model = gs_model  # Regression
        self.uq_model = uq_model  # UQ
        self.ds_model = ds_model  # Distance

        # Save location
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
                            self.splitter
                            )
        self.splits = list(splits)

    def split(self, X, y, g, splitter):

        # Train, test splits
        count = -1
        for split in splitter.split(X, y, g):
            train, test = split

            # Include some random test points.
            if self.sub_test > 0.0:
                sub = ShuffleSplit(n_splits=1, test_size=self.sub_test)
                sub = sub.split(X[train], y[train], g[train])
                sub = list(sub)[0]

                train = np.array(train)
                test = np.array(test)

                test = np.concatenate([test, train[sub[1]]])
                train = train[sub[0]]

            count += 1
            yield (train, test, count)

    def fit(self, split):

        gs_model = self.gs_model
        uq_model = self.uq_model
        ds_model = self.ds_model
        save = self.save

        train, test, count = split  # train/test

        # Fit models
        model = build_model(gs_model, ds_model, uq_model)
        data_cv = model.fit(self.X[train], self.y[train], self.g[train])
        data_test = model.predict(self.X[test])

        _, in_domain_test = ground_truth(
                                         self.y[test],
                                         data_test['y_pred'],
                                         model.cut,
                                         )

        data_test['y'] = self.y[test]
        data_test['g'] = self.g[test]
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'
        data_test['in_domain'] = in_domain_test

        data_cv['fold'] = count

        data = pd.concat([data_cv, data_test])
        data['index'] = data['index'].astype(int)

        return data

    def plot(self, df, mets, save):

        i, df = df
        mets = mets[(mets['split'] == i[0]) & (mets['fold'] == i[1])]

        # Plot ground truth
        job_name = list(map(str, i))
        job_name = os.path.join(*[save, job_name[0], job_name[1]])

        # Save locations
        sigma_name = os.path.join(job_name, 'sigma')
        dist_name = os.path.join(job_name, 'dissimilarity')

        plots.ground_truth(
                           df['y'],
                           df['y_pred'],
                           df['y_std'],
                           df['in_domain'],
                           job_name
                           )

        # Precision recall for in domain
        for i in [True, False]:
            if i is True:
                j = 'id'
            else:
                j = 'od'
            sigma_thresh = plots.pr(
                                    df['y_std'],
                                    df['in_domain'],
                                    i,
                                    os.path.join(sigma_name, j),
                                    choice='rel_f1',
                                    )

            dist_thresh = plots.pr(
                                   df['dist'],
                                   df['in_domain'],
                                   i,
                                   os.path.join(dist_name, j),
                                   choice='rel_f1',
                                   )

        # Plot prediction time
        std = df['y'].std()
        plots.assessment(
                         df['y_std'],
                         std,
                         df['y_std']/std,
                         df['in_domain'],
                         sigma_name,
                         )

        plots.assessment(
                         df['y_std'],
                         std,
                         df['dist'],
                         df['in_domain'],
                         dist_name,
                         )

        plots.violin(df['dist'], df['in_domain'], dist_name)
        plots.violin(df['dist'], df['in_domain'], sigma_name)

        # Total
        for i, j in zip(
                        ['in_domain_pred', 'out_domain_pred'],
                        ['id', 'od'],
                        ):

            for k in ['dist', 'y_std']:
                for w in [dist_name, sigma_name]:
                    plots.confusion(
                                    df['in_domain'],
                                    y_pred=df[k+'_'+i],
                                    pos_label=j,
                                    save=os.path.join(w, j)
                                    )

        # Plot CDF comparison
        res = df['y']-df['y_pred']
        absres = abs(res)/df['y_std']
        x = (res)/df['y_std']
        plots.cdf_parity(
                         x,
                         df['in_domain'],
                         save=job_name
                         )

        # Plot parity
        plots.parity(
                     mets,
                     df['y'].values,
                     df['y_pred'].values,
                     df['in_domain'].values,
                     save=job_name
                     )

        plots.violin(absres, df['in_domain'], save=job_name)

    def save_model(self):
        '''
        Build one model on all data.
        '''

        gs_model = self.gs_model
        uq_model = self.uq_model
        ds_model = self.ds_model
        save = self.save

        # Build the model
        model = build_model(gs_model, ds_model, uq_model)
        data_cv = model.fit(self.X, self.y, self.g)
        data_cv['fold'] = 0
        data_cv['split'] = 'cv'
        data_cv['index'] = data_cv['index'].astype(int)

        # Statistics
        print('Assessing CV statistics from data used for fitting')
        mets = group_metrics(data_cv, ['split', 'fold', 'in_domain'])

        # Save location
        original_loc = os.path.join(save, 'model')
        os.makedirs(original_loc, exist_ok=True)

        # Plot assessment
        print('Plotting results for CV splits: {}'.format(save))
        parallel(
                 self.plot,
                 data_cv.groupby(['split', 'fold']),
                 mets=mets,
                 save=original_loc,
                 )

        # Save the model
        dill.dump(model, open(os.path.join(original_loc, 'model.dill'), 'wb'))

        # Data
        pd.DataFrame(self.X).to_csv(os.path.join(
                                                 original_loc,
                                                 'X.csv'
                                                 ), index=False)
        X_trans = transforms(model.gs_model, self.X)
        pd.DataFrame(X_trans).to_csv(os.path.join(
                                                  original_loc,
                                                  'X_transformed.csv'
                                                  ), index=False)
        pd.DataFrame(self.y).to_csv(os.path.join(
                                                 original_loc,
                                                 'y.csv'
                                                 ), index=False)
        pd.DataFrame(self.g).to_csv(os.path.join(
                                                 original_loc,
                                                 'g.csv'
                                                 ), index=False)

        if hasattr(model.ds_model, 'bw'):
            bw = model.ds_model.bw
            np.savetxt(os.path.join(
                                    original_loc,
                                    'bw.csv'
                                    ), [bw], delimiter=',')

        data_cv.to_csv(os.path.join(
                                    original_loc,
                                    'train.csv'
                                    ), index=False)

    def assess(self):

        gs_model = self.gs_model
        uq_model = self.uq_model
        ds_model = self.ds_model
        save = self.save

        print('Assessing splits with ML pipeline: {}'.format(save))
        data = parallel(
                        self.fit,
                        self.splits,
                        )

        data = pd.concat(data)

        # Statistics
        print('Assessing test and CV statistics from data used for fitting')
        mets = group_metrics(data, ['split', 'fold', 'in_domain'])

        # Save locations
        assessment_loc = os.path.join(save, 'assessment')
        os.makedirs(assessment_loc, exist_ok=True)

        # Plot assessment
        print('Plotting results for test and CV splits: {}'.format(save))
        parallel(
                 self.plot,
                 data.groupby(['split', 'fold']),
                 mets=mets,
                 save=assessment_loc,
                 )

        # Save csv
        data.to_csv(os.path.join(
                                 assessment_loc,
                                 'assessment.csv'
                                 ), index=False)
