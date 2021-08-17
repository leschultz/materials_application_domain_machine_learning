import pandas as pd
import pkg_resources
import os

data_path = pkg_resources.resource_filename('mad', 'datasets/data')


def load(df, target, drop_cols=None):
    '''
    Returns data for regression task
    '''

    path = os.path.join(data_path, df)

    if '.csv' == df[-4:]:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)

    # Prepare data
    X = df.drop(target, axis=1)
    X_names = X.columns.tolist()
    X = X.values
    y = df[target].values

    data = {}
    data['data'] = X
    data['target'] = y
    data['feature_names'] = X_names
    data['target_name'] = target
    data['data_filename'] = path
    data['frame'] = df

    return data


def friedman():
    '''
    Load the Friedman dataset.
    '''

    # Dataset information
    df = 'friedman_data.csv'
    target = 'y'

    return load(df, target)


def super_cond():
    '''
    Load the super conductor data set.
    '''

    # Dataset information
    df = 'Supercon_data_features_selected.xlsx'
    target = 'Tc'
    drop_cols = [
                 'name',
                 'group',
                 'ln(Tc)',
                 ]

    return load(df, target, drop_cols)


def diffusion():
    '''
    Load the diffusion data set.
    '''

    # Dataset information
    df = 'Diffusion_Data_haijinlogfeaturesnobarrier_alldata.xlsx'
    target = 'E_regression'
    drop_cols = [
                 'Material compositions 1',
                 'Material compositions 2',
                 'E_regression_shift',
                 ]

    return load(df, target, drop_cols)


def test():
    '''
    Load the test dataset.
    '''

    # Dataset information
    df = 'test_data.csv'
    target = 'y'

    return load(df, target)
