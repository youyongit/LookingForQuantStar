# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha22(bars,fast=False):
    """Alpha22 of 101 Alphas"""
    open = bars['open']
    close = bars['open']
    high = bars['high']
    volume = bars['volume']
    df = correlation(high, volume, 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return -1 * delta(df, 5) * (stddev(close, 20))

    
    
