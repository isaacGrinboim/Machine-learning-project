from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        distances = cdist(X, self.X_train)
        indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self.y_train[indices]
        predictions = np.sign(np.sum(neighbors_labels, axis=1))
        return predictions