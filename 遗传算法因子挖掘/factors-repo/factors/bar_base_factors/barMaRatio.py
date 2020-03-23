# encoding: UTF-8
# fields = ['open','high','low','close','turnover','volume','openInterest']
from talib import ADX
import pandas as pd
import numpy as np
#from ctaFunction import ma_ratio

#----------------------------------------------------------------------
def barMaRatio(bars, timeperiod=5):
    """Ratio of diff between last price and mean value to last price """

    close = bars['close']
    result = close.rolling(timeperiod).apply(ma_ratio)

    return result