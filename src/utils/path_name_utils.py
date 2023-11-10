

import os


def get_model_save_path(cfg):
    return cfg.model_save_path

def get_model_file_name(cfg, epoch=None):
    parts = [cfg.fit.model, str(cfg.fit.time_length), str(cfg.fit.img_size)]

    if epoch is not None:
        parts.append(str(epoch))
    else:
        parts.append(str(cfg.fit.train.epochs))

    if cfg.fit.test.cal_type == "PEAK":
        parts.append(str(cfg.fit.test.cal_type))
    return '_'.join(parts) + '.pt'


def get_model_save_path_and_file_name(cfg):
    return os.path.join(get_model_save_path(cfg), get_model_file_name(cfg))

def get_dataset_directory(cfg):
    parts = [cfg.fit.dataset_name, str(cfg.fit.img_size), str(cfg.preprocess.larger_box_coef).replace('.', '_'), str(cfg.fit.time_length)]
    return os.path.join(cfg.dataset_path, 'segments', '_'.join(parts))

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from rppg.config import get_config
    cfg_path = '../configs/base_config.yaml'
    cfg = get_config(cfg_path)
    print(get_model_save_path_and_file_name(cfg))
    print(get_dataset_directory(cfg))


