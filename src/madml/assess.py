from madml.calculators import bin_data, ground_truth
from madml.models import assign_ground_truth
from madml.hosting import docker
from madml.plots import plotter
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

        # If user defined
        self.gt_rmse = self.model.gt_rmse
        self.gt_area = self.model.gt_area

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

    def cv(self, split, save_inner_folds=None):
        '''
        Fit models and get predictions from one split.
        '''

        train, test, count, name = split  # train/test

        if (train.shape[0] < 1) | (test.shape[0] < 1):
            return pd.DataFrame()

        # Fit models
        model = copy.deepcopy(self.model)
        model.fit(self.X[train], self.y[train], self.g[train])

        # Save fold fit
        if save_inner_folds is not None:
            save = os.path.join(save_inner_folds, 'train_folds')
            save = os.path.join(save, 'split_{}'.format(name))
            save = os.path.join(save, 'fold_{}'.format(count))
            os.makedirs(save, exist_ok=True)
            model.plot(save)

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

        # Ground truths
        data['gt_rmse'] = model.gt_rmse
        data['gt_area'] = model.gt_area

        return data

    def test(
             self,
             save_inner_folds=None,
             save_outer_folds=None,
             name=None,
             push_container=False,
             ):
        '''
        Gather assessment data and plot results.

        inputs:
            save_inner_folds = Top directory to save training folds.
            save_outer_folds = The top directory to save assessment.
            name = The name of the container <account>/<repository_name>:<tag>
            push_container = Whether to build and push a container with model.

        '''

        # Assess model
        df = []
        for i in tqdm(self.splits):
            df.append(self.cv(i, save_inner_folds))

        df = pd.concat(df)  # Combine data

        # Acquire ground truths
        self = ground_truth(self, self.y)
        df['gt_rmse'] = self.gt_rmse
        df['gt_area'] = self.gt_area

        # Ground truths
        df, df_bin = bin_data(df, self.model.bins)
        df, df_bin = assign_ground_truth(
                                         df,
                                         df_bin,
                                         )

        # Full fit
        self.model.fit(self.X, self.y, self.g)

        # Refit on out-of-bag data for final model
        self.model.domain_rmse.fit(
                                   df['d_pred_max'].values,
                                   df['domain_rmse/std_y'].values,
                                   )
        self.model.domain_area.fit(
                                   df['d_pred_max'].values,
                                   df['domain_cdf_area'].values,
                                   )

        if save_outer_folds is not None:

            # Save locations
            ass_save = os.path.join(save_outer_folds, 'assessment')
            model_save = os.path.join(save_outer_folds, 'model')

            # Create locations
            for d in [ass_save, model_save]:
                os.makedirs(d, exist_ok=True)

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
                           df,
                           df_bin,
                           self.model.precs,
                           ass_save,
                           )
            plot.generate()

            if name is not None:

                # Create a container
                data_path = pkg_resources.resource_filename(
                                                            'madml',
                                                            'templates/docker',
                                                            )
                save = os.path.join(save_outer_folds, 'hosting')
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

        return df, df_bin, self.model
