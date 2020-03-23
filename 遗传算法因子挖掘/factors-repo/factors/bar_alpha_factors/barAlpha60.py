# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha60(bars,fast=False):
    """Alpha60 of 101 Alphas"""
    close = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']
    divisor = (high - low).replace(0, 0.0001)
    inner = ((close - low) - (high - close)) * volume / divisor
    return - (2 * inner -ts_argmax(close, 10))
