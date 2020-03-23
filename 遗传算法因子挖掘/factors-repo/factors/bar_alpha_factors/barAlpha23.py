# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha23(bars,fast=False):
    """Alpha23 of 101 Alphas"""
    close = bars['close']
    high = bars['high']
    cond = sma(high, 20) < high
    alpha = np.zeros_like(close)
    alpha[cond] = -1 * delta(high, 2)[cond]
    return alpha
