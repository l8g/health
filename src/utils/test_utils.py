
import os

import pandas as pd


def save_single_result(result_path, result, cfg):
    csv_file = 'result.csv'
    if cfg.fit.test.cal_type == "PEAK":
        idx = '_'.join([cfg.fit.model, str(cfg.fit.time_length), str(cfg.fit.img_size), str(cfg.fit.test.cal_type)
                        , str(cfg.fit.train.epochs), str(cfg.fit.test.eval_time_length)])
    else:
        idx = '_'.join([cfg.fit.model, str(cfg.fit.time_length), str(cfg.fit.img_size), str(cfg.fit.train.epochs), 
                        str(cfg.fit.test.eval_time_length)])
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.isfile(result_path + csv_file):
        remaining_result = pd.read_csv(result_path + csv_file, index_col=0)
        if idx in remaining_result.index:
            print("Warning: The result allready exists, overwriting...")
            remaining_result = remaining_result.drop(idx)
        new_result = pd.DataFrame(columns=cfg.fit.test.metrics, index=[idx])
        new_result[cfg.fit.test.metrics] = result
        merged_result = pd.concat([remaining_result, new_result]).sort_index()
        merged_result.to_csv(result_path + csv_file)
    else:
        new_result = pd.DataFrame(columns=cfg.fit.test.metrics, index=[idx])
        new_result[cfg.fit.test.metrics] = result
        new_result.to_csv(result_path + csv_file)

        