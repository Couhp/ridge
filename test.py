import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
from sklearn.linear_model import Ridge

norm_sum = []

class RidgeRegressor(object):
    
    def fit(self, X, Y, alpha=0):
        one_vec = np.ones(shape=(len(X), 1))    
        X = np.hstack((one_vec, X))
        # one = np.ones((X.shape[0], X.shape[1]))
        G = alpha * np.eye(len(X[0] +1))
        self.params = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X) + G),X.T), Y)
                  
    def predict(self, X):
        one_vec = np.ones(shape=(len(X), 1))    
        X = np.hstack((one_vec, X))
        return np.dot(X, self.params)

def norm (data,train_set=False) :
    data = np.array(data).T
    # print (data)
    if train_set :
        for column in data :
            #per field in data
            max = np.max(column)
            min = np.min(column)
            norm_sum.append([max,min])
            column = (column - min)/(max - min)
        return data.T
    else :
        for _ in range(len(data)) :
            column = data[_]
            max = norm_sum[_][0]
            min = norm_sum[_][1]
            column = (column - min)/(max - min)
        return data.T
    


if __name__ == '__main__':


    # X = np.array(genfromtxt('prostate-training-data.csv', delimiter=',', dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7)))
    # Y = genfromtxt('prostate-training-data.csv', delimiter=',', dtype=None, usecols=(8))

    # clf = Ridge(alpha=1.0)
    # clf.fit(X,Y)
    # test = np.array(genfromtxt('test.csv', delimiter=',', dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7)))
    # print (clf.predict(test))
    
    X = np.array(genfromtxt('prostate-training-data.csv', delimiter=',', dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7)))
    X = norm(X, train_set=True)
    y = genfromtxt('prostate-training-data.csv', delimiter=',', dtype=None, usecols=(8))

    
    r = RidgeRegressor()
    r.fit(X, y)
    
    test = np.array(genfromtxt('test.csv', delimiter=',', dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7)))
    test = norm(test)
    print (r.predict(test))
    