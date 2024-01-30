from madml.models import assign_ground_truth
from madml.calculators import cdf, bin_data
from madml.hosting import docker
from madml.plots import plotter
from sklearn import metrics
from tqdm import tqdm

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
                 ):

        '''
        A class to split data into multiple levels.

        inputs:
            model = The combined model to assess.
            X = The original features to be split.
            y = The original target features to be split.
            g = The groups of data to be split.
            splitters = All the types of splitters to assess.
        '''

        self.X = X  # Features
        self.y = y  # Target
        self.splitters = copy.deepcopy(splitters)  # Splitter
        self.model = copy.deepcopy(model)

        self.model.disable_tqdm = True  # Disable print

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
        data['std_y'] = np.std(self.y[train])

        # Predictions
        data['r'] = self.y[test]-data['y_pred']
        data['z'] = data['r']/data['y_stdc_pred']
        data['r/std_y'] = data['r']/data['std_y']
        data['y_stdc_pred/std_y'] = data['y_stdc_pred']/data['std_y']

        # Get naive predictions
        data['naive_y'] = np.mean(self.y[train])
        pred = model.predict(self.X[train])
        data['naive_stdu'] = np.mean(pred['y_stdu_pred'])
        data['naive_stdc'] = np.mean(pred['y_stdc_pred'])

        return data

    def test(self):
        '''
        Gather assessment data and plot results.
        '''

        # Assess model
        df = []
        for i in tqdm(self.splits):
            df.append(self.cv(i))

        df = pd.concat(df)  # Combine data
        df,  df_bin = bin_data(df, self.model.bins)

        # Determine ground truth from test data
        df_id = df[df['splitter'] == 'fit']

        # Acquire ground truths
        if self.model.gt_rmse is None:
            true = df_id['y']/df_id['std_y']
            pred = df_id['naive_y']/df_id['std_y']
            rmse = metrics.mean_squared_error(
                                              true,
                                              pred,
                                              )
            rmse **= 0.5
            self.gt_rmse = rmse

        else:
            self.gt_rmse = self.model.gt_rmse

        if self.model.gt_area is None:
            pred = df_id['y']-df_id['y_pred']
            pred /= df_id['naive_stdc']
            self.gt_area = cdf(pred)[3]

        else:
            self.gt_area = self.model.gt_area

        # Classify ground truth labels
        assign_ground_truth(
                            df,
                            df_bin,
                            self.gt_rmse,
                            self.gt_area,
                            )

        self.df = df
        self.df_bin = df_bin

        # Full fit
        self.model.fit(self.X, self.y, self.g)

        return df, df_bin, self.model

    def write_results(self, parent, name=None, push_container=False):
        '''
        Push docker container with full fit model.

        inputs:
            parent = The top directory to save things.
            name = The name of the container <account>/<repository_name>:<tag>
            push_container = Whether to build and push a container with model.
        '''

        # Save locations
        ass_save = os.path.join(parent, 'assessment')
        model_save = os.path.join(parent, 'model')
        model_ass = os.path.join(model_save, 'train')

        # Create locations
        for d in [ass_save, model_save, model_ass]:
            os.makedirs(d, exist_ok=True)

        # Write test data
        self.df.to_csv(os.path.join(ass_save, 'pred.csv'), index=False)
        self.df_bin.to_csv(os.path.join(
                                        ass_save,
                                        'pred_bins.csv',
                                        ), index=False)

        # Save model
        np.savetxt(
                   os.path.join(model_save, 'X.csv'),
                   self.X,
                   delimiter=',',
                   )

        np.savetxt(
                   os.path.join(model_save, 'y.csv'),
                   self.y,
                   delimiter=',',
                   )

        np.savetxt(
                   os.path.join(model_save, 'g.csv'),
                   self.g,
                   delimiter=',',
                   fmt='%s',
                   )

        dill.dump(
                  self.model,
                  open(os.path.join(model_save, 'model.dill'), 'wb')
                  )

        # Test data
        plot = plotter(
                       self.df,
                       self.df_bin,
                       self.gt_rmse,
                       self.gt_area,
                       self.model.precs,
                       ass_save,
                       )
        plot.generate()

        # Train data
        plot = plotter(
                       self.model.data_cv,
                       self.model.bin_cv,
                       self.model.gt_rmse,
                       self.model.gt_area,
                       self.model.precs,
                       model_ass,
                       )
        plot.generate()

        if name is not None:

            # Create a container
            data_path = pkg_resources.resource_filename(
                                                        'madml',
                                                        'templates/docker',
                                                        )
            save = os.path.join(parent, 'hosting')
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
