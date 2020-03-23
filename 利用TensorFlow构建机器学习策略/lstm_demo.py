import sys,os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

sys.path.append('/home/jaq/project/simpleback')
sys_path = "/home/jaq/project/simpleback/factors_repo"

#os.chdir(project_base_path)
sys.path.append(sys_path)

print(sys.path)

import time
import json

# from distributed import Client
# client = Client(n_workers=4)
#import modin.pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
import numba

import datetime as dt
import scipy.stats as st
import matplotlib.pyplot as plt

from multiprocessing import Pool

from factors_repo import Factors
from common.get_data import get_data_df, load_kline_data

from sklearn.model_selection import train_test_split

from model import Model

DATASET = []

def make_data_set(df, dataframe_length, sample_length=14):
    X = []

    try:
        for idx in tqdm(range(dataframe_length)):
            if dataframe_length - idx >= sample_length:
                if idx == 0:
                    data_array = [df.iloc[idx: idx + sample_length].to_numpy().transpose()]
                else:
                    array_tmp = [df.iloc[idx: idx + sample_length].to_numpy().transpose()]
                    data_array = np.row_stack((data_array, array_tmp))
    except:
        print('error')

    return data_array
    
def profit_label(bars, sample_length=14, fast=False):
    """计算label profit"""

    close = bars['close'].iloc[sample_length - 1:].values
    x = close
    x_roll = np.roll(close, 1)
    x = x - x_roll

    x = pd.Series(x)

    return x 

def build_dataset(data_array, factors_name, label):
    factors = {}
    for idx, fi in enumerate(factors_name):
        factors[fi] = data_array[:, idx].reshape((len(data_array), 14, 1))

    dataset = tf.data.Dataset.from_tensor_slices((factors, label))
    dataset = dataset.batch(2).repeat()

    return dataset 

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

if __name__ == "__main__":

    SYMBOL = 'LTC'
    PERIOD = '60min'

    # 导入数据
    kline_df = load_kline_data(SYMBOL, PERIOD)
    kline_df = kline_df[kline_df['volume'] != 0]

    # 添加因子
    factors_manager = Factors()
    factors_manager.add_factor('barRSI_14', timeperiod=14)
    factors_manager.add_factor('barRSI_7', timeperiod=7)

    factors_name = ['open', 'high', 'low', 'close', 'volume']
    factors_name.extend(list(factors_manager.factors_info.keys()))

    # 计算因子
    factors_df = factors_manager.cal_candle_factors(kline_df)
    factors_df = factors_df.dropna().replace(np.inf, 9999).replace(-np.inf, 9999)

    # 计算label
    label = profit_label(factors_df).values

    # 按照数据长度分割数据
    dataframe_length = len(factors_df)
    data_array = make_data_set(factors_df, dataframe_length, sample_length=14)

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    X_train, X_test, y_train, y_test = train_test_split(data_array, label, test_size=0.2)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)

    train_dataset = build_dataset(X_train, factors_name, y_train)
    test_dataset = build_dataset(X_test, factors_name, y_test)
    valid_dataset = build_dataset(X_valid, factors_name, y_valid)

    print('dataset input len: {}'.format(len(data_array)))
    print('dataset label len: {}'.format(len(label)))

    # 建立模型
    model = Model(FACTORS_NAME=factors_name, INPUT_SHAPE=14)
    model.build_lstm()

    model.train(train_dataset, test_dataset)

    y_hat = model.predict(valid_dataset)

    show_plot((y_test[0:10],y_hat[0:10]), 0, 'Sample Example')
