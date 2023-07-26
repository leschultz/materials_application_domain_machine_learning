from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline

from madml.ml.splitters import BootstrappedLeaveClusterOut
from madml.models.space import distance_model
from madml.models.combine import domain_model
from madml.models.uq import calibration_model
from madml.ml.assessment import nested_cv
from madml import datasets


def keras_model(shape):

    model = Sequential()
    model.add(Dense(
                    1024,
                    input_dim=shape,
                    kernel_initializer='normal',
                    activation='relu'
                    ))
    model.add(Dropout(0.3))
    model.add(Dense(
                    1024,
                    kernel_initializer='normal',
                    activation='relu'
                    ))
    model.add(Dropout(0.3))
    model.add(Dense(
                    1,
                    kernel_initializer='normal'
                    ))
    model.compile(
                  loss='mean_squared_error',
                  optimizer='adam'
                  )

    return model


def main():

    run_name = 'run'
    data_name = replace_data

    # Load data
    data = datasets.load(data_name)
    X = data['data']
    y = data['target']
    g = data['class_name']
    n_repeats = 1

    # ML Distance model
    ds_model = distance_model(dist='kde')

    # ML UQ function
    uq_model = calibration_model(params=[0.0, 1.0])

    # ML
    scale = StandardScaler()
    model = KerasRegressor(
                           build_fn=keras_model,
                           shape=X.shape[1],
                           epochs=250,
                           batch_size=100,
                           verbose=0
                           )
    model = BaggingRegressor(model)

    # The grid for grid search
    grid = {}
    grid['model__n_estimators'] = [100]

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
                            )

    # Types of sampling to test
    splits = [('calibration', RepeatedKFold(n_repeats=n_repeats))]

    # Boostrap, cluster data, and generate splits
    for i in [2, 3]:

        # Cluster Splits
        top_split = BootstrappedLeaveClusterOut(
                                                AgglomerativeClustering,
                                                n_repeats=n_repeats,
                                                n_clusters=i
                                                )

        splits.append(('agglo_{}'.format(i), top_split))

    # Fit models
    model = domain_model(gs_model, ds_model, uq_model, splits)
    cv = nested_cv(X, y, g, model, splits, save=run_name)
    cv.assess()
    cv.push('leschultz/cmg:{}'.format(data_name))


if __name__ == '__main__':
    main()
