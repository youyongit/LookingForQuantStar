# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
import pandas as pd
import numpy as np
#from ctaFunction import *

#---------------------------------------------------------------
def barAlpha16(bars,fast=False):
    """Alpha16 of 101 Alphas"""
    high = bars['high']
    volume = bars['volume']
    return -1 * (covariance(high, volume, 5))
    
