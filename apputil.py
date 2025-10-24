import pandas as pd


def GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ["median", "mean"]:
            raise ValueError(f"You passed {estimate}. Estimate argument must be either median or mean") 
        self.estimate = estimate
    
    def fit(self, X, y):
        return None

    def predict(self, X):
        return None