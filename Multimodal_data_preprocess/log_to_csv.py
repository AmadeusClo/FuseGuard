import pandas as pd
import json
import os
import pytz
import re
import time
from datetime import datetime
import tarfile
import math

tz = pytz.timezone('Asia/Shanghai')
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def logs_to_csv(log_data_list, csv_file_path):
    data_csvs = pd.DataFrame(columns=['date', 'service', 'log_content'])
    for service,log_list in log_data_list.items():
        data_csv = pd.DataFrame(columns=['date', 'log_content'])
        for log in log_list:
            # log_datetime = ' '.join(log.split(' ')[:2])
            # 提取方括号内的日期和时间部分
            log_datetime_str = log.split(']')[0].strip('[')
            # log_datetime = pd.to_datetime(log_datetime, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            log_datetime = pd.to_datetime(log_datetime_str, format='%Y-%b-%d %H:%M:%S.%f', errors='coerce')
            # log_datetime = log_datetime.tz_localize('UTC').tz_convert('Asia/Shanghai')
            log_timestamp = math.floor(log_datetime.timestamp()) - 8 * 3600
            # log_content = log
            log_content = log.split(']')[1].strip()
            # data_csvs = data_csvs._append({'date': log_timestamp, 'service': service, 'log_content': log_content}, ignore_index=True)
            data_csv = data_csv._append({'date': log_timestamp, 'log_content': log_content}, ignore_index=True)
        service_csv_path = os.path.join(csv_file_path, f'{service}_logs.csv')
        data_csv.to_csv(service_csv_path, index=False)
    # data_csvs.to_csv(csv_file_path, index=True)
    return data_csvs


# 解压函数
def extract_all_tar_xz(tar_file_path, log_data_dir):
    # Ensure the target directory exists
    if not os.path.exists(log_data_dir):
        os.makedirs(log_data_dir)

    # Iterate through all files in the tar_file_path directory
    for filename in os.listdir(tar_file_path):
        if filename.endswith('.tar.xz'):
            # Construct the full path to the .tar.xz file
            full_tar_path = os.path.join(tar_file_path, filename)

            # Extract the .tar.xz file
            with tarfile.open(full_tar_path, 'r:xz') as tar:
                tar.extractall(path=log_data_dir)
                print(f"Extracted {full_tar_path} to {log_data_dir}")


# Paths
tar_file_path = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\no fault'

log_data_dir = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\data\SN'
log_data_process = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\data\Services'

# log_data_dir = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\no fault\SN'
# log_data_process = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\no fault\Services'

# Extract all .tar.xz files
# extract_all_tar_xz(tar_file_path, log_data_dir)

dataset = ['SN']

for ds in dataset:
    sn_logs_data_concat = {"compose-post-service": [], "home-timeline-service": [], "media-service": [],
                           "nginx-web-server": [], "post-storage-service": [], "social-graph-service": [],
                           "text-service": [],
                           "unique-id-service": [], "url-shorten-service": [], "user-mention-service": [],
                           "user-service": [],
                           "user-timeline-service": []}
    tt_logs_data_concat = {"ts-assurance-service": [], "ts-auth-service": [], "ts-basic-service": [],
                           "ts-cancel-service": [], "ts-config-service": [], "ts-contacts-service": [],
                           "ts-food-map-service": [],"ts-food-service": [], "ts-inside-payment-service": [], "ts-notification-service": [],
                           "ts-order-other-service": [],"ts-order-service": [],"ts-payment-service": [], "ts-preserve-service": [], "ts-price-service": [],
                           "ts-route-plan-service": [],"ts-route-service": [],
                           "ts-seat-service": [], "ts-security-service": [], "ts-station-service": [],
                           "ts-ticketinfo-service": [], "ts-train-service": [],
                           "ts-travel2-service": [], "ts-travel-plan-service": [],"ts-travel-service": [],
                           "ts-user-service": [], "ts-verification-code-service": []
                           }
    # json_file_path = os.path.join(log_data_dir, ds, f'{ds}_logall.json')
    # for path in os.listdir(os.path.join(log_data_dir, ds)):
    #     json_file_path = os.path.join(log_data_dir, ds, path, 'logs.json')  #每个span.json文件的路径
    #     if not os.path.exists(json_file_path):
    #         continue
    # 遍历目标目录下的所有文件
    for folder_name in os.listdir(log_data_dir):
        folder_path = os.path.join(log_data_dir, folder_name)
        if os.path.isdir(folder_path):  # 检查是否是文件夹
            json_file_path = os.path.join(folder_path, 'logs.json')
            if ds == 'SN':
                sn_log_data = read_json(json_file_path)
                for service,sn_log in sn_log_data.items():
                    servicename = service
                    sn_logs_data_concat[service].extend(sn_log)

        if ds == 'TT':
            tt_log_data = read_json(json_file_path)
            for tservice,tt_log in tt_log_data.items():
                servicename = tservice
                tt_logs_data_concat[tservice].extend(tt_log)

    if ds == 'SN':
        with open(os.path.join(log_data_dir, f'{ds}_all_logs.json'), 'w') as f:
            json.dump(sn_logs_data_concat, f)
    if ds == 'TT':
        with open(os.path.join(log_data_dir, ds, f'{ds}_all_logs.json'), 'w') as f:
            json.dump(tt_logs_data_concat, f)

# 将os.path.join(log_data_dir, f'{ds}_all_logs.json')转为csv
sn_logs_data_concat = read_json(os.path.join(log_data_dir, f'{ds}_all_logs.json'))
# sn_logs_data_concat = logs_to_csv(sn_logs_data_concat, os.path.join(log_data_dir, f'{ds}_all_logs.csv'))
sn_logs_data_concat = logs_to_csv(sn_logs_data_concat, log_data_process)

# 读取csv文件
sn_logs_data = read_csv(os.path.join(log_data_dir, f'{ds}_all_logs.csv'))
error_counts = sn_logs_data['log_content'].value_counts()
# 统计每个错误的出现次数
error_counts = error_counts.reset_index()
error_counts.columns = ['log_content', 'count']
# 将统计结果保存到CSV文件
error_counts.to_csv(os.path.join(log_data_dir, f'{ds}_error_counts.csv'), index=False)



