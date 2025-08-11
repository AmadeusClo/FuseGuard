from data_provider.data_loader import Dataset_Trace, Dataset_Trace_log, Dataset_Trace_log_metric
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'Trace': Dataset_Trace,
    'Trace_Log': Dataset_Trace_log,
    'Trace_Log_Metric': Dataset_Trace_log_metric,

}


def data_provider(args, flag, vali=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        metric_traindata_path=args.metric_traindata_path,
        trace_traindata_path=args.trace_traindata_path,
        log__traindata_path=args.log_traindata_path,
        metric_testdata_path=args.metric_testdata_path,
        trace_testdata_path=args.trace_testdata_path,
        log_testdata_path=args.log_testdata_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        timeenc=timeenc,
        freq=freq,
        percent=args.percent
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
