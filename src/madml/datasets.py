from sklearn import datasets as sklearn_datasets
from madml import datasets
import pandas as pd
import numpy as np

import pkg_resources
import os

data_path = pkg_resources.resource_filename('madml', 'data')


def loader(df, target, drop_cols=None, class_name=None, n=None, frac=None):
    '''
    Returns data.

    inputs:
        df = The path of the data.
        target = The name of the target variable.
        drop_cols = Columns to drop.
        class_name = The column containing groups.
        n = The number of samples to acquire from data.
        frac = The fraction of samples to acquire from data.
    '''

    path = os.path.join(data_path, df)
    data = {}

    if '.csv' == df[-4:]:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine='openpyxl')

    # Sub sample for testing
    if n:
        df = df.sample(n)
    elif frac:
        df = df.sample(frac=frac)

    if class_name:
        data['class_name'] = df[class_name].values
    else:
        data['class_name'] = np.repeat('none', df.shape[0])

    if drop_cols:
        data['dropped'] = df[drop_cols]
        df.drop(drop_cols, axis=1, inplace=True)

    # Prepare data
    X = df.drop(target, axis=1)
    X_names = X.columns.tolist()
    X = X.values
    y = df[target].values

    data['data'] = X
    data['target'] = y
    data['feature_names'] = X_names
    data['target_name'] = target
    data['data_filename'] = path
    data['frame'] = df

    return data


def load(name, *args, **kwargs):

    if name == 'friedman1':
        '''
        Load the Friedman1 dataset.
        '''

        # Dataset information
        n_samples = 500
        n_features = 5

        g = [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]

        gs = []
        for i in g:

            # Data are sampled from [start, end)
            X = np.random.uniform(
                                  low=i[0],
                                  high=i[1],
                                  size=(n_samples, n_features)
                                  )
            y = (
                 10*np.sin(np.pi*X[:, 0]*X[:, 1])
                 + 20*(X[:, 2]-0.5)**2
                 + 10*X[:, 3]
                 + 5*X[:, 4]
                 )

            if i[0] == 0.0:
                Xs = X
                ys = y
            else:
                Xs = np.concatenate((Xs, X))
                ys = np.concatenate((ys, y))

            gs += ['[{},{})'.format(*i)]*y.shape[0]

        X, y = sklearn_datasets.make_friedman1(n_samples, n_features)
        gs += ['[0.0,1.0)']*y.shape[0]

        Xs = np.concatenate((Xs, X))
        ys = np.concatenate((ys, y))

        X = Xs
        y = ys
        g = np.array(gs)

        df = pd.DataFrame(X)
        df['y'] = y
        df['g'] = g

        data = {}
        data['data'] = X
        data['target'] = y
        data['class_name'] = g
        data['feature_names'] = range(X.shape[0])
        data['target_name'] = 'y'
        data['data_filename'] = 'None'
        data['frame'] = df

        return data

    elif name == 'make_regression':
        '''
        Load the make_regression dataset.
        '''

        n_samples = 500
        n_features = 5

        g = [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]

        gs = []
        for i in g:

            # Data are sampled from [start, end)
            X = np.random.uniform(
                                  low=i[0],
                                  high=i[1],
                                  size=(n_samples, n_features)
                                  )

            if i[0] == 0.0:
                Xs = X
            else:
                Xs = np.concatenate((Xs, X))

            gs += ['[{},{})'.format(*i)]*X.shape[0]

        X = np.random.uniform(
                              low=0.0,
                              high=1.0,
                              size=(n_samples, n_features)
                              )

        Xs = np.concatenate((Xs, X))

        ground_truth = np.zeros((n_features, 1))
        ground_truth[:n_features, :] = 100*np.random.uniform(size=(
                                                                   n_features,
                                                                   1
                                                                   ))

        gs += ['[0.0,1.0)']*X.shape[0]

        X = Xs
        y = np.dot(X, ground_truth)
        y = np.squeeze(y)

        g = np.array(gs)

        df = pd.DataFrame(X)
        df['y'] = y
        df['g'] = g

        data = {}
        data['data'] = X
        data['target'] = y
        data['class_name'] = g
        data['feature_names'] = range(X.shape[0])
        data['target_name'] = 'y'
        data['data_filename'] = 'None'
        data['frame'] = df

        return data

    elif name == 'super_cond':
        '''
        Load the super conductor data set.
        '''

        # Dataset information
        df = 'Supercon_data_features_selected.xlsx'
        target = 'Tc'
        class_name = 'group'
        drop_cols = [
                     'name',
                     'group',
                     'ln(Tc)',
                     ]

        return loader(df, target, drop_cols, class_name, **kwargs)

    elif name == 'diffusion':
        '''
        Load the diffusion data set.
        '''

        # Dataset information
        df = 'diffusion.csv'
        target = 'E_regression_shift'
        class_name = 'group'
        drop_cols = [
                     'mat',
                     'group'
                     ]

        return loader(df, target, drop_cols, class_name, **kwargs)

    elif name == 'diffusion_all_features':
        '''
        Load the diffusion data set.
        '''

        # Dataset information
        df = 'diffusion_all_features.csv'
        target = 'E_regression_shift'
        class_name = 'group'
        drop_cols = [
                     'mat',
                     'group'
                     ]

        return loader(df, target, drop_cols, class_name, **kwargs)

    elif name == 'perovskite_stability':
        '''
        Load the perovskite stability dataset.
        '''

        df = 'Perovskite_stability_Wei_updated_forGlenn.xlsx'
        target = 'EnergyAboveHull'

        return loader(df, target, **kwargs)

    elif name == 'electromigration':
        '''
        Load the electronmigration dataset.
        '''

        df = 'Dataset_electromigration.xlsx'
        target = 'Effective_charge_regression'

        return loader(df, target, **kwargs)

    elif name == 'thermal_conductivity':
        '''
        Load the thermal conductivity dataset.
        '''

        df = 'citrine_thermal_conductivity_simplified.xlsx'
        target = 'log(k)'

        return loader(df, target, **kwargs)

    elif name == 'dielectric_constant':
        '''
        Load the dielectric constant dataset.
        '''

        df = 'dielectric_constant_simplified.xlsx'
        target = 'log(poly_total)'

        return loader(df, target, **kwargs)

    elif name == 'double_perovskites_gap':
        '''
        Load the double perovskie gap dataset.
        '''

        df = 'double_perovskites_gap.xlsx'
        target = 'gap gllbsc'

        return loader(df, target, **kwargs)

    elif name == 'perovskites_opband':
        '''
        Load the perovskie ipband dataset.
        '''

        df = 'Dataset_Perovskite_Opband_simplified.xlsx'
        target = 'O pband (eV) (GGA)'

        return loader(df, target, **kwargs)

    elif name == 'elastic_tensor':
        '''
        Load the elastic tensor dataset.
        '''

        df = 'elastic_tensor_2015_simplified.xlsx'
        target = 'log(K_VRH)'

        return loader(df, target, **kwargs)

    elif name == 'heusler_magnetic':
        '''
        Load the heussler magnetic dataset.
        '''

        df = 'heusler_magnetic_simplified.xlsx'
        target = 'mu_b saturation'

        return loader(df, target, **kwargs)

    elif name == 'piezoelectric_tensor':
        '''
        Load the piezoelectric tensor data.
        '''

        df = 'piezoelectric_tensor.xlsx'
        target = 'log(eij_max)'

        return loader(df, target, **kwargs)

    elif name == 'steel_yield_strength':
        '''
        Load the steel yield strength dataset.
        '''

        df = 'steel_strength.csv'
        target = 'yield_strength'
        drop_cols = [
                     'mat',
                     'group'
                     ]

        return loader(df, target, drop_cols, **kwargs)

    elif name == 'steel_yield_strength_all_features':
        '''
        Load the steel yield strength dataset.
        '''

        df = 'steel_strength_all_features.csv'
        target = 'yield_strength'
        drop_cols = [
                     'mat',
                     'group'
                     ]

        return loader(df, target, drop_cols, **kwargs)

    elif name == 'fluence':
        '''
        Load the steel fluence dataset.
        '''

        df = 'fluence.csv'
        target = 'Measured_DT41J_[C]'
        drop_cols = [
                     'group'
                     ]

        return loader(df, target, drop_cols, **kwargs)


def list_data():
    '''
    Return the list of data supported.
    '''

    datanames = [
                 'super_cond',
                 'diffusion',
                 'friedman1',
                 'steel_yield_strength',
                 'fluence',
                 'make_regression',
                 'fetch_california_housing',
                 ]

    return datanames
