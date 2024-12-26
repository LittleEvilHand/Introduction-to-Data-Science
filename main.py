import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
# from double import lstm
from LSTM2 import lstm_model

rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用 'SimHei'
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

for i in range(3,13):
    lstm_model(i, 120, 50)
