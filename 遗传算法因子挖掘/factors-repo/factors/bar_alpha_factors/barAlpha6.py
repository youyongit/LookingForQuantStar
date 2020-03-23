# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha6(bars,fast=False):
    """Alpha6 of 101 Alphas"""
    open = bars['open']
    volume = bars['volume']
    df = -1 * correlation(open, volume, 10)
    return df.replace([-np.inf, np.inf], 0).fillna(value=0)
