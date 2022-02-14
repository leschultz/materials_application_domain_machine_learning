from mad.functions import parallel
from joblib import load

import pandas as pd
import numpy as np
import glob


class model:

    def __init__(self, save):

        models = glob.glob(save+'/splits/model_*')
        uqfuncs = glob.glob(save+'/splits/uqfunc_*')
        distfuncs = glob.glob(save+'/splits/distfunc_*')

        self.models = parallel(load, models)
        self.uqfuncs = parallel(load, uqfuncs)
        self.distfuncs = parallel(load, distfuncs)

    def get_std_dist(self, X, items):

        pipe = items[0]
        uqfunc = items[1]
        distfunc = items[2]

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

        dist = distfunc.predict(X)
        dist = pd.DataFrame(dist)

        return std, dist

    def predict(self, X):

        y_pred = parallel(lambda i: i.predict(X), self.models)
        std_dist = parallel(
                            lambda i: self.get_std_dist(X, i),
                            list(zip(
                                     self.models,
                                     self.uqfuncs,
                                     self.distfuncs
                                     ))
                            )

        std_pred = [i[0] for i in std_dist]
        dist_pred = [i[1] for i in std_dist]
        dists = dist_pred[0].columns

        # Combine predictions
        df = []
        for i, j, k in zip(y_pred, std_pred, dist_pred):

            data = {}
            data['y_pred'] = i
            data['std_pred'] = j
            dist_pred = pd.DataFrame(k, columns=dists)

            data = pd.DataFrame(data)
            data = pd.concat([data, dist_pred], axis=1)
            data['index'] = data.index
            df.append(data)

        df = pd.concat(df)

        return df
