from madml.hosting import docker
from madml import plots

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
                 X,
                 y,
                 g=None,
                 model=None,
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
        self.splitters = splitters  # Splitter
        self.model = model
        self.save = save

        # Grouping
        if g is None:
            self.g = np.array(['no-groups']*self.X.shape[0])
        else:
            self.g = g

        # Generate the splits
        splits = self.split()
        self.splits = list(splits)

    def split(self):
        '''
        Generate the splits.
        '''

        # Train, test splits
        for splitter in self.splitters:
            for count, split in enumerate(splitter[1].split(
                                                            self.X,
                                                            self.y,
                                                            self.g
                                                            )):
                train, test = split
                yield (train, test, count, splitter[0])

    def cv(self, split):
        '''
        Fit models and get predictions from one split.
        '''

        train, test, count, name = split  # train/test

        # Fit models
        print('MADMl - Nested CV {} Fold: {}'.format(name, count))
        model = copy.deepcopy(self.model)
        model.fit(self.X[train], self.y[train], self.g[train])
        data_test = model.predict(self.X[test])

        data_test['y'] = self.y[test]
        data_test['g'] = self.g[test]
        data_test['std(y)'] = model.ystd
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'
        data_test['splitter'] = name
        data_test['index'] = data_test['index'].astype(int)
        data_test['r'] = data_test['y']-data_test['y_pred']

        # z score
        if 'y_stdu' in data_test.columns:
            data_test['z'] = data_test['r']/data_test['y_stdc']

            calc = data_test['y_stdc']/data_test['std(y)']
            data_test['y_stdc/std(y)'] = calc

            calc = data_test['y_stdu']/data_test['std(y)']
            data_test['y_stdu/std(y)'] = calc

        # Normalized
        data_test['r/std(y)'] = data_test['r']/data_test['std(y)']
        data_test['y_pred/std(y)'] = data_test['y_pred']/data_test['std(y)']
        data_test['y/std(y)'] = data_test['y']/data_test['std(y)']

        return data_test

    def assess(self):
        '''
        Gather assessment data and plot results.
        '''

        # Assess model
        df = [self.cv(i) for i in self.splits]
        df = pd.concat(df)
        df['id'] = abs(df['r/std(y)']) < 1.0

        save = os.path.join(self.save, 'assessment')
        out = plots.generate_plots(
                                   df,
                                   np.std(self.y),
                                   self.model.bins,
                                   save,
                                   )

        df.to_csv(os.path.join(save, 'single.csv'), index=False)

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
            name = The name of the container <account>/<repository_name>/<tag>
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
            print('MADML -Pushing model')
            docker.build_and_push_container(name)

        with open('user_predict.py', 'r') as handle:
            data = handle.read()

        data = data.replace('replace', name)

        with open('user_predict.py', 'w') as handle:
            handle.write(data)

        os.chdir(old)
