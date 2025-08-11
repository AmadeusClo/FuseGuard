import numpy as np
from tick.hawkes import HawkesADM4


def logHaw(chunk_logs, end_time, event_num, decay=3, ini_intensity=0.2):
    model = HawkesADM4(decay)
    model.fit(chunk_logs, end_time, baseline_start=np.ones(event_num) * ini_intensity)
    return np.array(model.baseline)


from util import read_json, Info
import pandas as pd
import os
import pickle


def deal_logs(intervals, info, idx, name):
    print("*** Dealing with logs...")

    df = pd.read_csv(os.path.join("./parsed_data", name, "logs" + idx + ".csv"))
    templates = read_json(os.path.join("./parsed_data", name, "templates.json"))
    event_num = len(templates)

    print("# Real Template Number:", event_num)
    event_num += 1  # 0: unseen

    event2id = {temp: idx + 1 for idx, temp in enumerate(templates)}  # 0: unseen
    event2id["Unseen"] = 0
    res = np.zeros((len(intervals), info.node_num, event_num))

    no_log_chunk = 0
    for chunk_idx, (s, e) in enumerate(intervals):
        if (chunk_idx + 1) % 100 == 0:
            print("Computing Hawkes of chunk {}/{}".format(chunk_idx + 1, len(intervals)))
        try:
            rows = df.loc[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
        except:
            no_log_chunk += 1
            continue

        service_events = rows.groupby("service")
        for service, sgroup in service_events:
            events = sgroup.groupby("event")
            knots = [np.array([0.0]) for _ in range(event_num)]
            for event, egroup in events:
                eid = 0 if event not in event2id else event2id[event]
                tmp = np.array(sorted(egroup["timestamp"].values)) - s
                adds = np.array([idx * (1e-5) for idx in range(len(tmp))])  # In case of too many identical numbers
                knots[eid] = tmp + adds
            paras = logHaw(knots, end_time=e + 1, event_num=event_num)
            res[chunk_idx, info.service2nid[service], :] = paras

    print("# Empty log:", no_log_chunk)
    with open(os.path.join("../chunks", name, idx, "logs.pkl"), "wb") as fw:
        pickle.dump(res, fw)
    return res


z_zero_scaler = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)


