# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *
#---------------------------------------------------------------
def barAlpha34(bars,fast=False):
    """Alpha34 of 101 Alphas"""
    close = bars['close']
    returns = bars['close']/bars['close'].shift(1)-1
    inner = stddev(returns, 2) / stddev(returns, 5)
    inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
    return (2 - (inner) - (delta(close, 1)))
