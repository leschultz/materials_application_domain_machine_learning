from sklearn.cluster import estimate_bandwidth
from scipy.spatial.distance import cdist

import numpy as np


class KDE:
    '''
    Much of the implementation was inspired by the package statsmodels.
    The implementation in this code expands the types of usable kernels.

    @inproceedings{seabold2010statsmodels,
      title={statsmodels: Econometric and statistical modeling with python},
      author={Seabold, Skipper and Perktold, Josef},
      booktitle={9th Python in Science Conference},
      year={2010},
    }
    '''

    def __init__(self, bandwidths, kernel):

        self.bandwidths = bandwidths
        self.kernel_name = kernel

        # Select the kernel function
        if kernel == 'gaussian':
            self.kernel = self.gaussian
        elif kernel == 'epanechnikov':
            self.kernel = self.epanechnikov
        elif kernel == 'tophat':
            self.kernel = self.tophat
        elif kernel == 'linear':
            self.kernel = self.linear
        elif kernel == 'cosine':
            self.kernel = self.cosine
        elif kernel == 'exponential':
            self.kernel = self.exponential
        else:
            raise "Non supported kernel type."

    def gaussian(self, bandwidth, x_train_col, x):
        '''
        Gaussian kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        k = (2.0*np.pi)**-0.5
        k *= np.exp(-0.5*a**2.0)

        return k

    def epanechnikov(self, bandwidth, x_train_col, x):
        '''
        Epanechnikov kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        indx = (-1 <= a) & (a <= 1)

        k = np.zeros_like(x_train_col)
        k[indx] = 0.75*(1.0-a[indx]**2.0)

        return k

    def tophat(self, bandwidth, x_train_col, x):
        '''
        Tophat kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        indx = (-1 <= a) & (a <= 1)

        k = np.zeros_like(x_train_col)
        k[indx] = 0.5

        return k

    def linear(self, bandwidth, x_train_col, x):
        '''
        Triangular (linear) kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        indx = (-1 <= a) & (a <= 1)

        k = np.zeros_like(x_train_col)
        k[indx] = 1-np.abs(a[indx])

        return k

    def cosine(self, bandwidth, x_train_col, x):
        '''
        Cosine kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        indx = (-1 <= a) & (a <= 1)

        k = np.zeros_like(x_train_col)
        k[indx] = 0.25*np.pi*np.cos(0.5*np.pi*a[indx])

        return k

    def exponential(self, bandwidth, x_train_col, x):
        '''
        Exponential kernel function.
        '''

        a = x_train_col-x
        a /= bandwidth

        k = 0.5*np.exp(-np.abs(a))

        return k

    def fit(self, X_train):
        '''
        Store the data that the KDE space is defined in.
        '''

        self.X_train = X_train

    def compute(self, x_row):
        '''
        Calculate the kernel of a sample's ith element to the training
        space's ith dimension.
        '''

        K = np.empty(self.X_train.shape)
        for i in range(self.X_train.shape[1]):
            K[:, i] = self.kernel(
                                  self.bandwidths[i],
                                  self.X_train[:, i],
                                  x_row[i],
                                  )

        K = K.prod(axis=1)/np.prod(self.bandwidths)
        K = K.sum()
        K /= self.X_train.shape[0]

        return K

    def score(self, X):
        '''
        Compute the probability density function (PDF) estimate
        for each case.
        '''

        pdfs = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            pdfs[i] = self.compute(X[i, :])

        return pdfs

    def predict(self, X):
        '''
        Transform the outputs to conform to a standard.
        '''

        pdfs = self.score(X)
        pdfs *= -1.0

        return pdfs


class distance_model:

    def __init__(
                 self,
                 dist='kde',
                 kernel='epanechnikov',
                 bandwidth=None,
                 weights=None,
                 ):

        self.dist = dist
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.weights = weights

    def fit(
            self,
            X_train,
            ):
        '''
        Get the distances based on a metric.

        inputs:
            X_train = The features of the training set.
            dist = The distance to consider.
            model = The model to get feature importances.
            n_features = The number of features to keep.

        ouputs:
            dists = A dictionary of distances.
        '''

        if self.dist == 'kde':

            if self.bandwidth is None:
                self.bandwidth = estimate_bandwidth(X_train)

            if self.bandwidth > 0.0:

                bandwidths = np.repeat(
                                       self.bandwidth,
                                       X_train.shape[1],
                                       )
                model = KDE(
                            kernel=self.kernel,
                            bandwidths=bandwidths,
                            )

                model.fit(X_train)

                dist = model.predict(X_train)
                m = np.max(dist)

                def pred(X):
                    out = model.predict(X)
                    return out

                self.model = pred

            else:
                self.model = lambda x: np.repeat(0.0, len(x))

        else:

            def pred(X):
                out = cdist(X_train, X, self.dist)
                out = np.mean(out, axis=0)
                return out

            self.model = pred

    def predict(self, X):
        '''
        Get the distances for individual cases.

        inputs:
            X = The features of data.
        outputs:
            dist = The distance array.
        '''

        return self.model(X)
