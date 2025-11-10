import pandas as pd
import numpy as np


def GroupEstimate(object):
    def __init__(self, estimate):
        """Making sure estimate is either median or mean"""
        if estimate not in ["median", "mean"]:
            raise ValueError(f"You passed {estimate}. Estimate argument must be either median or mean") 
        self.estimate = estimate
    
    def fit(self, X, y):
        """Fit function from part two"""
        #Check that there are no missing values in y
        if any(np.isnan(y)): 
            raise ValueError(f"Array y contains missing values. Please correct data and try again") 
        #Check that x and y are the same size 
        if len(X) != len(y): 
            raise ValueError(f"Dataframe X is not the same lenght as array y") 
        #Get the variable names for X 
        cols = list(X.columns)
        #Combin X and y
        X["target"] = y

        #Group by the variables in X and provide estimate of them 
        est_dict = dict()
        for col in cols: 
            if self.estimate == "mean":
                est_dict[col] = X.groupby(col)['target'].mean()
            else: 
                est_dict[col] = X.groupby(col)['target'].mean()

        return est_dict

    def predict(self, X):
        return None