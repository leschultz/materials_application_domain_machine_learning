from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import inspect
from collections import defaultdict

class MLKR_KNN():
    def __init__(self, n_neighbors=5, n_components=None, max_iter=400, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        self._knn = KNeighborsRegressor(n_neighbors=n_neighbors if n_neighbors > 0 else 5, weights=weights,
                                        algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                                        metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._metric_learner = MLKR(init='auto', n_components=n_components, max_iter=max_iter, verbose=1)
        self._feature_importances_ = None
        self._find_k = False if n_neighbors > 0 else True
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.n_neighbors = n_neighbors
        self.n_components = n_neighbors
        self.max_iter = max_iter
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
        
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])


    @property
    def find_k(self):
        return self._find_k

    @property
    def knn(self):
        return self._knn

    def transform(self, X):
        return self._metric_learner.transform(X)

    def fit(self, X, y, **kwargs):
        self._metric_learner.fit(X, y)
        self._feature_importances_ = np.sum(self._metric_learner.components_.T, axis=1)
        self._feature_importances_ = self._feature_importances_ / np.sum(self._feature_importances_)
        fitted_X = self.transform(X)
        return self.knn.fit(fitted_X, y)

    def predict(self, X):
        fitted_X = self.transform(X)
        return self.knn.predict(fitted_X)

    @property
    def feature_importances_(self):
        return self._feature_importances_

