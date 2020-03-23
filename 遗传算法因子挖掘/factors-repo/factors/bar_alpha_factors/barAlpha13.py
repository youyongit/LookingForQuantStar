# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha13(bars,fast=False):
    """Alpha13 of 101 Alphas"""
    close = bars['close']
    volume = bars['volume']
    return -1 * (covariance(close, volume, 5))
