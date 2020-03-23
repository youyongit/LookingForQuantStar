# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha2(bars,fast=False):
    """Alpha2 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    volume = bars['volume']
    df = -1 * correlation(delta(log(volume), 2),(close-open)/open,6)
    return df.replace([-np.inf, np.inf], 0).fillna(value=0)

