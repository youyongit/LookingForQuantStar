# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha20(bars,fast=False):
    """Alpha20 of 101 Alphas"""
    close = bars['close']
    open = bars['open']
    low = bars['low']
    high = bars['high']
    return -1 * ((open - delay(high, 1)) *\
                     (open - delay(close, 1)) *\
                     (open - delay(low, 1)))
    
    
