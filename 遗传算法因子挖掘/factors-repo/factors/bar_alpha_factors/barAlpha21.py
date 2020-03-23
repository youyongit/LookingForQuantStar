# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha21(bars,fast=False):
    """Alpha21 of 101 Alphas"""
    close = bars['close']
    volume = bars['volume']
    cond_1 = sma(close, 8) + stddev(close, 8) < sma(close, 2)
    cond_2 = sma(volume, 20) / volume < 1
    alpha = np.ones_like(close)
    alpha[(cond_1 | cond_2)] = -1
    return alpha
    
    
