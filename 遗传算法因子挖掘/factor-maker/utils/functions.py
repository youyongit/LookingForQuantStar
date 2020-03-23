import numpy as np
import pandas as pd
from scipy.stats import rankdata
from gplearn.functions import make_function

def _rolling_rank(data):
    value = rankdata(data)[-1]
    
    return value

def _rolling_prod(data):
    
    return np.prod(data)

def _ts_sum(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
    value = np.nan_to_num(value)

    return value

def _sma(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
    value = np.nan_to_num(value)
    
    return value

def _stddev(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)
    
    return value

def _ts_rank(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_rank).tolist())
    value = np.nan_to_num(value)
    
    return value

def _product(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_prod).tolist())
    value = np.nan_to_num(value)
    
    return value

def _ts_min(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)
    
    return value

def _ts_max(data):
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)
    
    return value

def _delta(data):
    value = np.diff(data.flatten())
    value = np.append(0, value)

    return value

def _delay(data):
    period=1
    value = pd.Series(data.flatten()).shift(1)
    value = np.nan_to_num(value)
    
    return value

def _rank(data):
    value = np.array(pd.Series(data.flatten()).rank().tolist())
    value = np.nan_to_num(value)
    
    return value

def _scale(data):
    k=1
    data = pd.Series(data.flatten())
    value = data.mul(1).div(np.abs(data).sum())
    value = np.nan_to_num(value)
    
    return value

def _ts_argmax(data):
    window=10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmax) + 1 
    value = np.nan_to_num(value)
    
    return value

def _ts_argmin(data):
    window=10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmin) + 1 
    value = np.nan_to_num(value)
    
    return value

# make_function函数群
delta = make_function(function=_delta, name='delta', arity=1)
delay = make_function(function=_delay, name='delay', arity=1)
rank = make_function(function=_rank, name='rank', arity=1)
scale = make_function(function=_scale, name='scale', arity=1)
sma = make_function(function=_sma, name='sma', arity=1)
stddev = make_function(function=_stddev, name='stddev', arity=1)
product = make_function(function=_product, name='product', arity=1)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
ts_min = make_function(function=_ts_min, name='ts_min', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=1)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=1)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=1)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=1)

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
user_function = [delta, delay, rank, scale, sma, stddev, product, ts_rank, ts_min, ts_max, ts_argmax, ts_argmin, ts_sum]