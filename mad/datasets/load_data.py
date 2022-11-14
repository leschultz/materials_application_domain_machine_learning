import pandas as pd
import numpy as np

import pkg_resources
import os

data_path = pkg_resources.resource_filename('mad', 'datasets/data')


def load(df, target, drop_cols=None, class_name=None, n=None, frac=None):
    '''
    Returns data for regression task
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
        data['class_name'] = ['no-groups']*df.shape[0]

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


def friedman(**kwargs):
    '''
    Load the Friedman dataset.
    '''

    # Dataset information
    df = 'friedman_data.csv'
    target = 'y'

    return load(df, target, **kwargs)


def super_cond(**kwargs):
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

    return load(df, target, drop_cols, class_name, **kwargs)


def diffusion(**kwargs):
    '''
    Load the diffusion data set.
    '''

    # Dataset information
    df = 'Diffusion_Data.csv'
    target = 'E_regression_shift'
    class_name = 'group'
    drop_cols = [
                 'Material compositions 1',
                 'Material compositions 2',
                 'E_regression',
                 'group'
                 ]

    return load(df, target, drop_cols, class_name, **kwargs)


def perovskite_stability(**kwargs):
    '''
    Load the perovskite stability dataset.
    '''

    df = 'Perovskite_stability_Wei_updated_forGlenn.xlsx'
    target = 'EnergyAboveHull'

    return load(df, target, **kwargs)


def electromigration(**kwargs):
    '''
    Load the electronmigration dataset.
    '''

    df = 'Dataset_electromigration.xlsx'
    target = 'Effective_charge_regression'

    return load(df, target, **kwargs)


def thermal_conductivity(**kwargs):
    '''
    Load the thermal conductivity dataset.
    '''

    df = 'citrine_thermal_conductivity_simplified.xlsx'
    target = 'log(k)'

    return load(df, target, **kwargs)


def dielectric_constant(**kwargs):
    '''
    Load the dielectric constant dataset.
    '''

    df = 'dielectric_constant_simplified.xlsx'
    target = 'log(poly_total)'

    return load(df, target, **kwargs)


def double_perovskites_gap(**kwargs):
    '''
    Load the double perovskie gap dataset.
    '''

    df = 'double_perovskites_gap.xlsx'
    target = 'gap gllbsc'

    return load(df, target, **kwargs)


def perovskites_opband(**kwargs):
    '''
    Load the perovskie ipband dataset.
    '''

    df = 'Dataset_Perovskite_Opband_simplified.xlsx'
    target = 'O pband (eV) (GGA)'

    return load(df, target, **kwargs)


def elastic_tensor(**kwargs):
    '''
    Load the elastic tensor dataset.
    '''

    df = 'elastic_tensor_2015_simplified.xlsx'
    target = 'log(K_VRH)'

    return load(df, target, **kwargs)


def heusler_magnetic(**kwargs):
    '''
    Load the heussler magnetic dataset.
    '''

    df = 'heusler_magnetic_simplified.xlsx'
    target = 'mu_b saturation'

    return load(df, target, **kwargs)


def piezoelectric_tensor(**kwargs):
    '''
    Load the piezoelectric tensor data.
    '''

    df = 'piezoelectric_tensor.xlsx'
    target = 'log(eij_max)'

    return load(df, target, **kwargs)


def steel_yield_strength(**kwargs):
    '''
    Load the steel yield strenght dataset.
    '''

    df = 'steel_strength_simplified.xlsx'
    target = 'yield strength'

    return load(df, target, **kwargs)
