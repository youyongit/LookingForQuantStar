# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha28(bars,fast=False):
    """Alpha28 of 101 Alphas"""
    close = bars['close']
    low = bars['low']
    high = bars['high']
    volume = bars['volume']
    adv20 = sma(volume, 20)
    df = correlation(adv20, low, 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return df + (high + low) / 2 - close
