import numpy as np 
import pandas as pd 
#from ctaFunction import values_deviation

def barPriceDev(bars, timeperiod=5):

    close = bars['close']
    result = close.rolling(timeperiod).apply(values_deviation)

    return result

