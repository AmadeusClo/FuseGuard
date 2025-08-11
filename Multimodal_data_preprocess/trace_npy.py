import numpy as np
import os
import pandas as pd
#############################################进行metrics数据时间对齐#############################################

# df = pd.read_csv(r'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08\tracedata\timestamp_trace_raw\test_ab_1.csv')
df = pd.read_csv(r'C:\mym\Dataset\PEMS08\tracedata\timestamp_trace_raw\traindata_1.csv')

directory = r'C:\mym\Dataset\SN and TT\SN Dataset\SN Dataset\no fault\Services'
output_directory = r'C:\mym\Dataset\PEMS08\logs\normal'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)

        # 读取 nginx-web-server_all_metric.csv 文件
        nginx_df = pd.read_csv(file_path)

        # 根据 df 中的 start_time 列筛选 nginx_df 中的相同时间戳的数据
        filtered_nginx_df = nginx_df[nginx_df['date'].isin(df['start_time'])]

        # 生成一个新的 DataFrame，其中第一列是 df 中的 start_time 列，其他列是 filtered_nginx_df 中对应时间戳的数据
        new_df = pd.merge(df[['start_time']], filtered_nginx_df, left_on='start_time', right_on='date',
                          how='left').drop(columns=['date'])

        # 生成一个新的文件
        output_file_path = os.path.join(output_directory, f'filtered_{filename}')
        new_df.to_csv(output_file_path, index=False)


#############################################进行trace与metrics数据时间对齐#############################################

# # 读取 test_data
# train_data = pd.read_csv(r'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08\tracedata\test_ab_1.csv', header=None).values
#
# # 指定目录包含过滤后的指标文件
# directory = r'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08\tracedata\without_net\trace_metric_abnormal'
#
# # 初始化最终数据为 test_data
# train_data = train_data.reshape((2160, 5, 1))
# # train_data = train_data.reshape((2832, 5, 1))
#
# # 遍历过滤后的指标文件
# metric_data_list = []
#
# # 遍历过滤后的指标文件
# for filename in os.listdir(directory):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(directory, filename)
#
#         # 读取指标数据
#         metric_data = pd.read_csv(file_path, header=None).values
#
#         # 重塑指标数据为 (2831, 1, 7)
#         metric_data = metric_data.reshape((2160, 1, 3))
#         # metric_data = metric_data.reshape((2832, 1, 3))
#
#         # 将重塑后的指标数据添加到列表中
#         metric_data_list.append(metric_data)
#
# # 将所有的 metric_data 沿第二维度合并为 (2831, 5, 7)
# merge_data = np.concatenate(metric_data_list, axis=1)
#
# # 将 merge_data 和 train_data 沿第三维度合并为 (2831, 5, 8)
# final_data = np.concatenate((train_data, merge_data), axis=2)
# # 检查 final_data 是否存在空值
# if np.isnan(final_data).any():
#     print("final_data contains NaN values.")
#     nan_indices = np.argwhere(np.isnan(final_data))
#     print("Indices of NaN values:", nan_indices)
# else:
#     print("final_data does not contain any NaN values.")
#
# # 保存最终连接的数据为新的 .npy 文件
# np.save(r'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08\merge_flow_test_nonet.npy', final_data)

#############################################进行数据归一化处理#############################################
traindata = np.load('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/merge_flow_nonet.npy')
testdata = np.load('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/merge_flow_test_nonet.npy')

means = np.zeros((5, 4))
stds = np.zeros((5, 4))

# 对每个通道进行归一化处理
for node in range(5):
    for channel in range(4):
        means[node, channel] = np.mean(traindata[:, node, channel])
        stds[node, channel] = np.std(traindata[:, node, channel])

# 对每个节点的每个通道进行归一化处理
normalized_data = np.zeros_like(traindata)
for node in range(5):
    for channel in range(4):
        normalized_data[:, node, channel] = (traindata[:, node, channel] - means[node, channel]) / stds[node, channel]
    column_name = f'train_node_{node}'
#     保存归一化后的数据normalized_data[:, node, ：]为csv文件,此时csv的维度为(2831, 4)
    pd.DataFrame(normalized_data[:, node, :]).to_csv(f'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08/normalizad_data_input/normal/{column_name}.csv', index=False)


# 对测试数据进行归一化处理
normalized_testdata = np.zeros_like(testdata)
for node in range(5):
    for channel in range(4):
        normalized_testdata[:, node, channel] = (testdata[:, node, channel] - means[node, channel]) / stds[node, channel]
    column_name = f'test_node_{node}'
    pd.DataFrame(normalized_testdata[:, node, :]).to_csv(f'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08/normalizad_data_input/abnormal/{column_name}.csv', index=False)
# 保存归一化后的数据
np.save('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/normalized_flow.npy', normalized_data)
np.save('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/normalized_flow_test.npy', normalized_testdata)















#############################################进行异常npy文件的读取和保存#############################################
# 定义矩阵
# matrix = np.array([
#     [1, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1]
# ])
#
# # 保存为npy格式
# np.save('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/adj.npy', matrix)

data = np.loadtxt(r'D:\code\diffusionmodel\DiffSTG-main\data\dataset\PEMS08\tracedata\traindata.csv', delimiter=',')

# 调整数组形状
train_data = data.reshape((2833, 5, 1))
# test_data = data.reshape((2160, 5, 1))

# 保存为 npy 文件
np.save('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/flow_ab.npy', test_data)

data_read = np.expand_dims(np.load('D:\code\diffusionmodel\DiffSTG-main\data/dataset/PEMS08/flow_ab.npy')[:, :, 0], -1)


