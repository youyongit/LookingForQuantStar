import numpy as np 
import pandas as pd 
from gplearn.fitness import make_fitness

def _my_metric(y, y_pred, w):
    value = np.sum(np.abs(y) + np.abs(y_pred))

    return value

def _msle(y, y_pred, w):
    value = np.square((np.log1p(y) - np.log1p(y_pred))).mean()

    return value

def _mse(y, y_pred, w):
    value = np.square(np.subtract(y, y_pred)).mean() 

    return value

my_metric = make_fitness(function=_my_metric, greater_is_better=True)
MSLE = make_fitness(function=_msle, greater_is_better=False)
MSE = make_fitness(function=_mse, greater_is_better=False)