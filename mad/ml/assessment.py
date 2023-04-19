from sklearn.metrics import precision_recall_curve, mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.base import clone
from sklearn.svm import SVC

from mad.stats.group import stats, group_metrics
from mad.utils import parallel, find
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
                 sigma,
                 cut=1.0,
                 ):

    # Define ground truth
    absres = abs(y-y_pred)/sigma

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
    sigma_y = []
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

        std = std_pred(gs_model_cv, X_trans_te)

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

        sigma_y += [np.std(y[train][tr])]*len(y[train][te])

    data = pd.DataFrame()
    data['g'] = g_cv
    data['y'] = y_cv
    data['y_pred'] = y_cv_pred
    data['y_std'] = y_cv_std
    data['dist'] = dist_cv
    data['index'] = index_cv
    data['split'] = 'cv'
    data['sigma_y'] = sigma_y  # Of the data trained on

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

        # Define ground truth
        cut, in_domain = ground_truth(
                                      data_id['y'],
                                      data_id['y_pred'],
                                      data_id['sigma_y'],
                                      )

        self.cut = cut

        data_id['in_domain'] = in_domain

        # Leave out 2 clusters
        od2_split = splitters.RepeatedClusterSplit(
                                                   AgglomerativeClustering,
                                                   n_repeats=1,
                                                   n_clusters=2
                                                   )

        data_od2 = cv(
                      self.gs_model,
                      self.ds_model,
                      X,
                      y,
                      g,
                      np.arange(y.shape[0]),
                      od2_split
                      )

        # Define ground truth
        cut, in_domain = ground_truth(
                                      data_od2['y'],
                                      data_od2['y_pred'],
                                      data_od2['sigma_y'],
                                      )

        data_od2['in_domain'] = in_domain

        # Leave out 3 clusters
        od3_split = splitters.RepeatedClusterSplit(
                                                   AgglomerativeClustering,
                                                   n_repeats=1,
                                                   n_clusters=3
                                                   )

        data_od3 = cv(
                      self.gs_model,
                      self.ds_model,
                      X,
                      y,
                      g,
                      np.arange(y.shape[0]),
                      od3_split
                      )

        # Define ground truth
        cut, in_domain = ground_truth(
                                      data_od3['y'],
                                      data_od3['y_pred'],
                                      data_od3['sigma_y'],
                                      )

        data_od3['in_domain'] = in_domain

        data_cv = pd.concat([data_id, data_od2, data_od3])

        # Calibrate uncertainties
        data_cv['y_std'] = self.uq_model.predict(data_cv['y_std'])

        # Z scores
        data_cv['z'] = (data_cv['y']-data_cv['y_pred'])/data_cv['y_std']

        self.domain_cut = {'dist': {}, 'y_std': {}}
        for i in [True, False]:

            for j in ['dist', 'y_std']:

                self.domain_cut[j][i] = plots.pr(
                                                 data_cv[j],
                                                 data_cv['in_domain'],
                                                 pos_label=i,
                                                 choice='rel_f1',
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
                 save='.',
                 ):

        self.X = X  # Features
        self.y = y  # Target
        self.splitter = splitter  # Splitter
        self.sub_test = sub_test

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
                                         model.ystd,
                                         )

        z = (self.y[test]-data_test['y_pred'])/data_test['y_std']

        data_test['y'] = self.y[test]
        data_test['z'] = z
        data_test['g'] = self.g[test]
        data_test['sigma_y'] = model.ystd
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

        if isinstance(i, tuple):
            mets = mets[(mets['split'] == i[0]) & (mets['fold'] == i[1])]
            i = list(i)
        else:
            i = [i, 'aggregate']
            mets = mets[(mets['split'] == i[0])]

        # Plot ground truth
        job_name = list(map(str, i))
        job_name = os.path.join(*[save, *job_name])

        # Save locations
        sigma_name = os.path.join(job_name, 'sigma')
        dist_name = os.path.join(job_name, 'dissimilarity')

        plots.ground_truth(
                           df['y'],
                           df['y_pred'],
                           df['y_std'],
                           df['sigma_y'],
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
                                    'rel_f1',
                                    os.path.join(sigma_name, j),
                                    )

            dist_thresh = plots.pr(
                                   df['dist'],
                                   df['in_domain'],
                                   i,
                                   'rel_f1',
                                   os.path.join(dist_name, j),
                                   )

        # Plot prediction time
        res = abs(df['y']-df['y_pred'])
        plots.assessment(
                         res,
                         df['sigma_y'],
                         df['y_std']/df['sigma_y'],
                         df['in_domain'],
                         sigma_name,
                         )

        plots.assessment(
                         res,
                         df['sigma_y'],
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

            for k, w in zip(['dist', 'y_std'], [dist_name, sigma_name]):
                plots.confusion(
                                df['in_domain'],
                                y_pred=df[k+'_'+i],
                                pos_label=j,
                                save=os.path.join(w, j)
                                )

        # Plot CDF comparison
        plots.cdf_parity(
                         df['z'],
                         df['in_domain'],
                         save=job_name
                         )

        # Plot the confidence curve
        plots.intervals(
                        df[[
                            'z',
                            'dist',
                            'y',
                            'y_pred',
                            'y_std',
                            'sigma_y',
                            ]].copy(),
                        'dist',
                        save=dist_name
                        )

        plots.intervals(
                        df[[
                            'z',
                            'y_std',
                            'y',
                            'y_pred',
                            'sigma_y',
                            ]].copy(),
                        'y_std',
                        save=sigma_name
                        )

        # Plot parity
        plots.parity(
                     mets,
                     df['y'].values,
                     df['y_pred'].values,
                     df['in_domain'].values,
                     save=job_name
                     )

        plots.violin(res, df['in_domain'], save=job_name)

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

        # Now for aggregate assessment
        mets = group_metrics(data, ['split', 'in_domain'])
        parallel(
                 self.plot,
                 data.groupby('split'),
                 mets=mets,
                 save=assessment_loc,
                 )

    def aggregate(self, parent='.'):
        '''
        If other independend runs were ran, then aggreagate those
        results and make overall statistic.
        '''

        paths = find(parent, 'assessment.csv')

        data = []
        for i in paths:
            run = i.split('/')[1]
            i = pd.read_csv(i)
            i['run'] = run
            data.append(i)

        data = pd.concat(data)

        save = os.path.join(parent, 'aggregate')
        os.makedirs(save, exist_ok=True)
        data.to_csv(os.path.join(save, 'aggregate.csv'))

        mets = group_metrics(data, ['split', 'in_domain'])
        parallel(
                 self.plot,
                 data.groupby('split'),
                 mets=mets,
                 save=save,
                 )
