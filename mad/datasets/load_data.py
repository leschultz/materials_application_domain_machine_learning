import pandas as pd
import pkg_resources
import os

data_path = pkg_resources.resource_filename('mad', 'datasets/data')


def load(df, target):
    '''
    Returns data for regression task
    '''

    path = os.path.join(data_path, df)

    if '.csv' == df[-4:]:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

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

    return data


def friedman():
    '''
    Load the Friedman dataset.
    '''

    # Dataset information
    df = 'friedman_data.csv'
    target = 'y'

    return load(df, target)
