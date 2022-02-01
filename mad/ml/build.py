from mad.functions import parallel
from joblib import load

import numpy as np
import glob


class model:

    def __init__(self, save):

        models = glob.glob(save+'/splits/model_*')
        uqfuncs = glob.glob(save+'/splits/uqfunc_*')

        self.models = parallel(load, models)
        self.uqfuncs = parallel(load, uqfuncs)

    def get_std(self, X, items):

        pipe = items[0]
        uqfunc = items[1]

        pipe = pipe.best_estimator_
        scaler = pipe.named_steps['scaler']
        select = pipe.named_steps['select']
        model = pipe.named_steps['model']

        X = scaler.transform(X)
        X = select.transform(X)

        std = []
        for i in model.estimators_:
            std.append(i.predict(X))

        std = np.std(std, axis=0)
        std = uqfunc.predict(std)

        return std

    def predict(self, X):

        y_pred = parallel(lambda i: i.predict(X), self.models)
        std_pred = parallel(
                            lambda i: self.get_std(X, i),
                            list(zip(self.models, self.uqfuncs))
                            )

        # Combine predictions
        y_pred = np.mean(y_pred, axis=0)
        std_pred = np.mean(std_pred, axis=0)

        return y_pred, std_pred
