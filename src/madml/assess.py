from madml.models import bin_data, assign_ground_truth
from madml.hosting import docker
from madml import plots
from scipy import stats

import pandas as pd
import numpy as np

import pkg_resources
import subprocess
import shutil
import copy
import dill
import os


class nested_cv:
    '''
    Class to do nested CV.
    '''

    def __init__(
                 self,
                 model,
                 X,
                 y,
                 g=None,
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
        self.splitters = copy.deepcopy(splitters)  # Splitter
        self.model = copy.deepcopy(model)
        self.save = save

        if g is None:
            self.g = np.repeat('not_provided', y.shape[0])
        else:
            self.g = g

        # Train, test splits
        self.splits = []
        for splitter in self.splitters:
            for count, split in enumerate(splitter[1].split(
                                                            self.X,
                                                            self.y,
                                                            self.g
                                                            ), 1):
                train, test = split
                self.splits.append((train, test, count, splitter[0]))

    def cv(self, split):
        '''
        Fit models and get predictions from one split.
        '''

        train, test, count, name = split  # train/test

        if (train.shape[0] < 1) | (test.shape[0] < 1):
            return pd.DataFrame()

        # Fit models
        print('MADMl - Nested CV {} Fold: {}'.format(name, count))
        model = copy.deepcopy(self.model)
        model.fit(self.X[train], self.y[train], self.g[train])

        # Gather predictions on test set
        data = model.predict(self.X[test])

        # Starting values
        data['index'] = test.astype(int)
        data['splitter'] = name
        data['fold'] = count
        data['y'] = self.y[test]

        # Statistics from training data
        data['range_y'] = np.ptp(self.y[train])
        data['iqr_y'] = stats.iqr(self.y[train])
        data['var_y'] = np.var(self.y[train])
        data['std_y'] = np.std(self.y[train])
        data['mad_y'] = stats.median_abs_deviation(self.y[train])

        # Predictions
        data['r'] = self.y[test]-data['y_pred']
        data['z'] = data['r']/data['y_stdc_pred']
        data['r/std_y'] = data['r']/data['std_y']

        return data

    def test(self):
        '''
        Gather assessment data and plot results.
        '''

        # Assess model
        df = [self.cv(i) for i in self.splits]
        df = pd.concat(df)  # Combine data

        # Determine ground truth from test data
        df_id = df[df['splitter'] == 'fit']
        bin_id = bin_data(df_id, self.model.bins, 'd_pred')
        df_bin = bin_data(df, self.model.bins, 'd_pred')

        gt_rmse = bin_id['rmse/std_y'].max()
        gt_area = bin_id[bin_id['cdf_area'] < bin_id['cdf_area'].max()].max()
        gt_area = bin_id[bin_id['bin'] != 'last']['cdf_area'].max()

        # Classify ground truth labels
        assign_ground_truth(
                            df,
                            df_bin,
                            gt_rmse,
                            gt_area,
                            )

        save = os.path.join(self.save, 'assessment')
        out = plots.generate_plots(
                                   df,
                                   np.std(self.y),
                                   self.model.bins,
                                   save,
                                   self.model.gts,
                                   self.model.gtb,
                                   self.model.dists,
                                   )
        thresholds, df_bin = out
        for i in ['y_stdc/std(y)', 'dist']:
            iname = 'intervals_{}.csv'.format(i.replace('/', '_'))
            df_bin[i].to_csv(os.path.join(*[
                                            save,
                                            'intervals',
                                            iname,
                                            ]), index=False)

        # Some nan values from concatenating on thresholds that do not exist
        df.fillna(False, inplace=True)

        if save:
            plots.generate_confusion(df, df_bin, save)

        df.to_csv(os.path.join(*[save, 'single', 'single.csv']), index=False)

        # Save model
        print('MADML - Making Full Fit Model')
        self.model.save = os.path.join(self.save, 'model')
        self.model.fit(self.X, self.y, self.g)

        np.savetxt(
                   os.path.join(self.model.save, 'X.csv'),
                   self.X,
                   delimiter=',',
                   )

        np.savetxt(
                   os.path.join(self.model.save, 'y.csv'),
                   self.y,
                   delimiter=',',
                   )

        np.savetxt(
                   os.path.join(self.model.save, 'g.csv'),
                   self.g,
                   delimiter=',',
                   fmt='%s',
                   )

        dill.dump(
                  self.model,
                  open(os.path.join(self.model.save, 'model.dill'), 'wb')
                  )

        return df, self.model

    def push(self, name, push_container=False):
        '''
        Push docker container with full fit model.

        inputs:
            name = The name of the container <account>/<repository_name>:<tag>
            push_container = Whether to build and push a container with model.
        '''

        print('MADML - Creating files to upload model to {}'.format(name))
        data_path = pkg_resources.resource_filename(
                                                    'madml',
                                                    'templates/docker',
                                                    )
        save = os.path.join(self.save, 'hosting')
        shutil.copytree(data_path, save)

        old = os.getcwd()
        os.chdir(save)
        shutil.copy('../model/model.dill', '.')
        shutil.copy('../model/X.csv', '.')

        # Capture current environment
        env = subprocess.run(
                             ['pip', 'freeze'],
                             capture_output=True,
                             text=True
                             )

        with open('requirements.txt', 'w') as handle:
            handle.write(env.stdout)

        if push_container:
            print('MADML - Pushing model')
            docker.build_and_push_container(name)

        with open('user_predict.py', 'r') as handle:
            data = handle.read()

        data = data.replace('replace', name)

        with open('user_predict.py', 'w') as handle:
            handle.write(data)

        with open('user_predict.ipynb', 'r') as handle:
            data = handle.read()

        data = data.replace('replace', name)

        with open('user_predict.ipynb', 'w') as handle:
            handle.write(data)

        os.chdir(old)
