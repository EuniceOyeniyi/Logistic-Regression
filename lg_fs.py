import numpy as np

class LogisticRegression:
    def __init__(self, l_rate=0.01, n_iter=1000):
        self.lr = l_rate
        self.ni = n_iter