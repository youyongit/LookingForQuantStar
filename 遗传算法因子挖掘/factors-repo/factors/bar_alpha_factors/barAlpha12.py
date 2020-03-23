# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha12(bars,fast=False):
    """Alpha12 of 101 Alphas"""
    close = bars['close']
    volume = bars['volume']
    return sign(delta(volume, 1)) * (-1 * delta(close, 1))
