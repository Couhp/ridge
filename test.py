import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt


class RidgeRegressor(object):
    def fit(self, X, y, alpha=0):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # one = np.ones((X.shape[0], X.shape[1]))
        G = alpha * np.eye(X.shape[1])
        self.params = np.dot(np.linalg.pinv(np.dot(X.T, X) + G),
                             np.dot(X.T, y))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        return np.dot(X, self.params)


if __name__ == '__main__':
    X = genfromtxt('prostate-training-data.csv', delimiter=',', dtype=None, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    print(X)
    y = genfromtxt('prostate-training-data.csv', delimiter=',', dtype=None, usecols=(8))
    print(y)

    # Create feature matrix
    # tX = np.array([X]).T
    # tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))

    # Plot regressors
    r = RidgeRegressor()
    r.fit(X, y)
    plt.plot(X, r.predict(X), 'b', label=u'ŷ (alpha=0.0)')
    alpha = 3.0
    r.fit(X, y, alpha)
    plt.plot(X, r.predict(X), 'y', label=u'ŷ (alpha=%.1f)' % alpha)

    plt.legend()
    plt.show()