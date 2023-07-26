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
        data['class_name'] = np.array(['no-groups']*df.shape[0])

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
        Load the Friedman dataset.
        '''

        # Dataset information
        df = 'friedman1.csv'
        target = 'y'

        return loader(df, target)

    elif name == 'make_regression':
        '''
        Load the make_regression dataset.
        '''

        # Dataset information
        df = 'make_regression.csv'
        target = 'y'

        return loader(df, target)

    elif name == 'fetch_california_housing':
        '''
        Load the fetch_california_housing dataset.
        '''

        # Dataset information
        df = 'fetch_california_housing.csv'
        target = 'y'

        return loader(df, target)

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
