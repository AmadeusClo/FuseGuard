import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

class Dataset_Trace_log_metric(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', metric_traindata_path='metric_traindata.csv',trace_traindata_path='trace_traindata.csv',log_traindata_path='/log_traindata/',
                 metric_testdata_path='metric_testdata.csv',trace_testdata_path='trace_testdata.csv',log_testdata_path='/log_testdata/',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.metric_traindata_path = metric_traindata_path
        self.trace_traindata_path = trace_traindata_path
        self.log_traindata_path = log_traindata_path
        self.metric_testdata_path = metric_testdata_path
        self.trace_testdata_path = trace_testdata_path
        self.log_testdata_path = log_testdata_path

        if self.set_type == 0:
            self.__read_traindata__()
        else:
            self.__read_testdata__()

    def __read_traindata__(self):
        self.scaler = StandardScaler()
        df_raw_metric = pd.read_csv(os.path.join(self.root_path,
                                                self.metric_traindata_path))
        self.data_metric = df_raw_metric.iloc[:, 1:]

        df_raw_trace = pd.read_csv(os.path.join(self.root_path,
                                          self.trace_traindata_path))
        self.data_stamp = df_raw_trace[['start_time']]
        self.data_trace = df_raw_trace.iloc[:, 1:]

        self.data_log = {}
        directory = os.path.join(self.root_path, self.log_traindata_path)
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                key = os.path.splitext(filename)[0]
                self.data_log[key] = df


    def __read_testdata__(self):
        self.scaler = StandardScaler()
        df_raw_metric = pd.read_csv(os.path.join(self.root_path,
                                                 self.metric_testdata_path))
        self.data_metric = df_raw_metric.iloc[:, 1:]

        df_raw_trace = pd.read_csv(os.path.join(self.root_path,
                                                self.trace_testdata_path))
        self.data_stamp = df_raw_trace[['start_time']]
        self.data_trace = df_raw_trace.iloc[:, 1:]

        self.data_log = {}
        directory = os.path.join(self.root_path, self.log_testdata_path)
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                key = os.path.splitext(filename)[0]
                self.data_log[key] = df

    def process_log_data(self, begin_stamp, end_stamp):
        result_sequences = []
        begin_stamp = pd.to_datetime(begin_stamp.iloc[0], unit='s')  # Assuming it's a single value
        end_stamp = pd.to_datetime(end_stamp.iloc[0], unit='s')

        for key, log_df in self.data_log.items():
            log_df['date'] = pd.to_datetime(log_df['date'], unit='s')  # Assuming 'date' is in seconds since epoch

            x_filtered_logs = log_df[(log_df['date'] >= begin_stamp) & (log_df['date'] < end_stamp)]
            x_template_counts = x_filtered_logs['log_content'].value_counts()

            # 将x时间段的日志模板及频次组合成序列
            sequence = "[CLS]"
            for template, count in x_template_counts.items():
                sequence += f"[{template}] ({count}) "
            sequence += "[SEP]"
            result_sequences.append(sequence)

        return result_sequences

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if self.set_type == 0:
            s_begin = index
        elif self.set_type == 2:
            s_begin = index * (self.seq_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        if r_end > len(self.data_trace):
            raise IndexError("r_end exceeds the data length")
        if r_end > len(self.data_stamp):
            raise IndexError("r_end exceeds the data stamp length")

        seq_x_trace = self.data_trace[s_begin:s_end].values
        seq_y_trace = self.data_trace[r_begin:r_end].values

        seq_x_metric = self.data_metric[s_begin:s_end].values
        seq_y_metric = self.data_metric[r_begin:r_end].values

        x_begin_stamp = self.data_stamp.iloc[s_begin]
        x_end_stamp = self.data_stamp.iloc[s_end]
        y_begin_stamp = self.data_stamp.iloc[r_begin]
        # y_end_stamp = self.data_stamp.iloc[r_end]
        try:
            y_end_stamp = self.data_stamp.iloc[r_end-1]
        except IndexError as e:
            print(f"IndexError: {e}, r_end: {r_end}, data_stamp length: {len(self.data_stamp)}")
            raise

        seq_x_log = self.process_log_data(x_begin_stamp, x_end_stamp)
        seq_y_log = self.process_log_data(y_begin_stamp, y_end_stamp)


        return seq_x_trace, seq_y_trace, seq_x_log, seq_y_log, seq_x_metric, seq_y_metric

    def __len__(self):
        return len(self.data_trace) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Trace_log(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        if self.set_type == 0:
            self.__read_traindata__()
        else:
            self.__read_testdata__()

    def __read_traindata__(self):
        self.scaler = StandardScaler()
        df_raw_trace = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        self.data_stamp = df_raw_trace[['start_time']]
        self.data_trace = df_raw_trace.iloc[:, 1:]

        self.data_log = {}
        directory = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\no fault\Services'
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                key = os.path.splitext(filename)[0]
                self.data_log[key] = df


    def __read_testdata__(self):
        self.scaler = StandardScaler()
        df_raw_trace = pd.read_csv('C:\mym\CALF-main\CALF-main\datasets\ETT-small/testdata_trace_timestamp.csv')
        self.data_stamp = df_raw_trace[['start_time']]
        self.data_trace = df_raw_trace.iloc[:, 1:]

        self.data_log = {}
        directory = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\data\Services'
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                key = os.path.splitext(filename)[0]
                self.data_log[key] = df

    def process_log_data(self, begin_stamp, end_stamp):
        result_sequences = []
        begin_stamp = pd.to_datetime(begin_stamp.iloc[0], unit='s')  # Assuming it's a single value
        end_stamp = pd.to_datetime(end_stamp.iloc[0], unit='s')

        for key, log_df in self.data_log.items():
            log_df['date'] = pd.to_datetime(log_df['date'], unit='s')  # Assuming 'date' is in seconds since epoch

            x_filtered_logs = log_df[(log_df['date'] >= begin_stamp) & (log_df['date'] < end_stamp)]
            x_template_counts = x_filtered_logs['log_content'].value_counts()

            # 将x时间段的日志模板及频次组合成序列
            sequence = "[CLS]"
            for template, count in x_template_counts.items():
                sequence += f"[{template}] ({count}) "
            sequence += "[SEP]"
            result_sequences.append(sequence)

        return result_sequences

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if self.set_type == 0:
            s_begin = index
        elif self.set_type == 2:
            s_begin = index * (self.seq_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        if r_end > len(self.data_trace):
            raise IndexError("r_end exceeds the data length")
        if r_end > len(self.data_stamp):
            raise IndexError("r_end exceeds the data stamp length")

        seq_x_trace = self.data_trace[s_begin:s_end].values
        seq_y_trace = self.data_trace[r_begin:r_end].values

        x_begin_stamp = self.data_stamp.iloc[s_begin]
        x_end_stamp = self.data_stamp.iloc[s_end]
        y_begin_stamp = self.data_stamp.iloc[r_begin]
        # y_end_stamp = self.data_stamp.iloc[r_end]
        try:
            y_end_stamp = self.data_stamp.iloc[r_end-1]
        except IndexError as e:
            print(f"IndexError: {e}, r_end: {r_end}, data_stamp length: {len(self.data_stamp)}")
            raise

        seq_x_log = self.process_log_data(x_begin_stamp, x_end_stamp)
        seq_y_log = self.process_log_data(y_begin_stamp, y_end_stamp)


        return seq_x_trace, seq_y_trace, seq_x_log, seq_y_log

    def __len__(self):
        return len(self.data_trace) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Trace(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        if self.set_type == 0:
            self.__read_traindata__()
        else:
            self.__read_testdata__()

    def __read_traindata__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        self.data_x = df_raw
        self.data_y = df_raw
        self.data_stamp = df_raw

    def __read_testdata__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv('C:\mym\CALF-main\CALF-main\datasets\ETT-small/testdata_trace.csv')

        self.data_x = df_raw
        self.data_y = df_raw
        self.data_stamp = df_raw


    def __getitem__(self, index):
        if self.set_type == 0:
            s_begin = index
        elif self.set_type == 2:
            s_begin = index * (self.seq_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].values
        seq_y = self.data_y[r_begin:r_end].values
        seq_x_mark = self.data_stamp[s_begin:s_end].values
        seq_y_mark = self.data_stamp[r_begin:r_end].values


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Workload_Ali(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        total_length = len(df_raw)
        # 按照7:1:2的比例划分数据
        few_shot_train_length = int(total_length * 0.35)
        train_length = int(total_length * 0.7)
        val_length = int(total_length * 0.1)
        test_length = total_length - train_length - val_length

        border1s = [0, train_length, train_length + val_length]
        border2s = [train_length - self.seq_len, train_length + val_length - self.seq_len, total_length - self.seq_len]
        # border1s = [0, few_shot_train_length, train_length + val_length]
        # border2s = [few_shot_train_length - self.seq_len, train_length + val_length - self.seq_len, total_length - self.seq_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # 使用iloc获取第0列的第border1到border2行的数据
        df_stamp = df_raw.iloc[border1:border2, 0]

        self.data_x = df_raw.iloc[border1:border2].values  # 转换为numpy.ndarray
        self.data_y = df_raw.iloc[border1:border2].values  # 转换为numpy.ndarray
        self.data_stamp = df_stamp.to_frame().values  # 转换为numpy.ndarray

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

