from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from sklearn.svm import SVR


def return_model(name, X):

    if name == 'rf':
        return RandomForestRegressor(n_estimators=100)

    elif name == 'bols':
        return BaggingRegressor(LinearRegression(), n_estimators=100)

    elif name == 'bsvr':
        return BaggingRegressor(SVR(), n_estimators=100)

    elif name == 'bnn':
        model = KerasRegressor(
                               build_fn=keras_model,
                               shape=X.shape[1],
                               epochs=500,
                               batch_size=100,
                               verbose=0,
                               )

        return BaggingRegressor(model, n_estimators=10)

    else:
        raise 'No model matching name.'


def keras_model(shape):

    n = 100
    model = Sequential()
    model.add(Dense(
                    n,
                    input_dim=shape,
                    kernel_initializer='normal',
                    activation='relu'
                    ))
    model.add(Dropout(0.3))
    model.add(Dense(
                    n,
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
