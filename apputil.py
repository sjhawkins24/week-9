import pandas as pd
import numpy as np


class GroupEstimate:
    def __init__(self, estimate):
        """Making sure estimate is either median or mean"""
        if estimate not in ["median", "mean"]:
            raise ValueError(f"You passed {estimate}. Estimate argument must be either median or mean") 
        self.estimate = estimate
    
    def fit(self, X, y):
        """Fit function from part two"""
        #Check that there are no missing values in y
        print(X.columns)
        #print(y.columns)
        if any(np.isnan(y)): 
            raise ValueError(f"Array y contains missing values. Please correct data and try again") 
        #Check that x and y are the same size 
        if len(X) != len(y): 
            raise ValueError(f"Dataframe X is not the same lenght as array y") 
        #Get the variable names for X 
        cols = list(X.columns)
        print("Got Columns")
        #Combin X and y
        X["target"] = y
        print("Added y to x")

        #Group by the variables in X and provide estimate of them 
        if self.estimate == "mean":
            ests =  X.groupby(cols)['target'].mean()
        else: 
            ests =  X.groupby(cols)['target'].median()
        self.ests = ests
        return None

    def predict(self, X_):
        """Function for prediction exercise in part three"""
        #Start by getting the results of .fit
        #Now store it in a dataframe that we can use 
        data = pd.DataFrame(self.ests)

        #Now we use a for loop to work through the prediction values 
        res = []

        for row in X_:  
            #Create a mask 
            mask = (data.iloc[:, 0:(len(data.columns)-1)] == row)
            #Get the corresponding target value
            res.append(data["target"][mask.sum(axis=1) == 2])
        return res