# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha44(bars,fast=False):
    """Alpha44 of 101 Alphas"""
    high = bars['high']
    volume = bars['volume']
    df = correlation(high, volume, 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return -1 * df
