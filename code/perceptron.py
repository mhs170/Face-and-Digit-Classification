import numpy as np
import random

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
    
    def fit(self, X, y):

        # initializing the weights to 0
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])]
        self.errors_ = []

        print("Weights:", self.w_)

        # training the model n_iter times
        for _ in range(self.n_iter):
            error = 0

            # loop through each input
            for xi, y in zip(X, y):

                # 1. calculate ŷ (the predicted value)
                y_pred = self.predict(xi)

                # 2. calculate Update
                # update = η * (y - ŷ)
                update = self.eta * (y - y_pred)

                # 3. Update the weights
                # Wi = Wi + Δ(Wi)   where  Δ(Wi) = η * (y - ŷ) = update * Xi
                self.w_[1:] = self.w_[1:] + update * xi
                print("Updated Weights:", self.w_[1:])

                # Update the bias (Xo = 1)
                self.w_[0] = self.w_[0] + update

                # if update != 0, it means that ŷ != y
                error += int(update != 0.0)

            self.errors_.append(error)

        return self