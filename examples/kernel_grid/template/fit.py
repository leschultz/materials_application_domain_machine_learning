from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from madml.models import dissimilarity, calibration, combine
from madml.splitters import BootstrappedLeaveClusterOut
from madml.assess import nested_cv
from madml import datasets

from mods import return_model


def main():

    run_name = 'output'
    data_name = replace_data
    model = replace_model
    kernel = replace_kernel

    # Load data
    data = datasets.load(data_name)
    X = data['data']
    y = data['target']

    # MADML parameters
    n_repeats = 5
    bins = 10

    # ML Distance model
    ds_model = dissimilarity('kde', kernel)

    # ML UQ function
    uq_model = calibration(params=[0.0, 1.0])

    # ML
    scale = StandardScaler()
    model = return_model(model, X)

    # The grid for grid search
    grid = {}

    # The machine learning pipeline
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('model', model),
                           ])

    # The gridsearch model
    gs_model = GridSearchCV(
                            pipe,
                            grid,
                            cv=((slice(None), slice(None)),),  # No splits
                            scoring='neg_mean_squared_error',
                            )

    # Types of sampling to test
    splits = [('fit', RepeatedKFold(n_repeats=n_repeats, n_splits=5))]

    # Boostrap, cluster data, and generate splits
    for i in [2, 3]:

        # Cluster Splits
        top_split = BootstrappedLeaveClusterOut(
                                                AgglomerativeClustering,
                                                n_repeats=n_repeats,
                                                n_clusters=i
                                                )

        splits.append(('agglo_{}'.format(i), top_split))

    # Assess models
    model = combine(gs_model, ds_model, uq_model, splits, bins=bins)
    cv = nested_cv(model, X, y, splitters=splits)
    df, df_bin, fit_model = cv.test(
                                    save_outer_folds=run_name,
                                    )


if __name__ == '__main__':
    main()
