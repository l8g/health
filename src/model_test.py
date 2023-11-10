
from rppg.config import get_config
from utils.path_name_utils import get_model_save_path_and_file_name


cfg_path = './configs/base_config.yaml'
cfg = get_config(cfg_path)

model_path_name = get_model_save_path_and_file_name(cfg)

print(model_path_name)
