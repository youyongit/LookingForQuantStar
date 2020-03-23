# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha18(bars,fast=False):
    """Alpha18 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    df = correlation(close, open, 10)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return -1 * ((stddev(abs((close - open)), 5) + (close - open)) + df)
    
    
