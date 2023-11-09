import random
import sys
import numpy as np
import torch
from dataset_loader import dataset_loader, data_loader
from dataset_loader2 import create_dataloaders, create_splits, get_file_paths
from loss import loss_fn
from models import get_model
from optim import optimizer

from rppg.config import get_config
from rppg.datasets.SegmentDataset import SegmentDataset
from run import run
from utils.test_utils import save_single_result


SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

generator = torch.Generator()
generator.manual_seed(SEED)



if __name__ == '__main__':
    cfg = get_config('./configs/base_config.yaml')
    result_save_path = 'result/csv/'
    directory = cfg.dataset_path + '/segments/UBFC_128_1_5_180'
    batch_size = cfg.fit.train.batch_size

        
    # 获取文件路径
    file_paths = get_file_paths(directory)
    # file_paths = file_paths[:10]

    # 分割数据集
    train_files, val_files, test_files = create_splits(file_paths)

    # 根据文件分割创建各自的SegmentDataset实例
    train_dataset = SegmentDataset(train_files)
    val_dataset = SegmentDataset(val_files)
    test_dataset = SegmentDataset(test_files)

    # 创建DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    model = get_model(fit_cfg=cfg.fit)

    opt = optimizer(model_params=model.parameters(), 
                        leanring_rate=cfg.fit.train.learning_rate, 
                        optim=cfg.fit.train.optimizer)
    criterion = loss_fn(loss_name=cfg.fit.train.loss)
    lr_sch = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=cfg.fit.train.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=cfg.fit.train.epochs
    )

    test_result = run(model=model, 
                      optimizer=opt, 
                      lr_sch=lr_sch, 
                      criterion=criterion, 
                      cfg=cfg, 
                      dataloaders=(train_loader, val_loader, test_loader))
    
    save_single_result(result_path=result_save_path, 
                       result=test_result, 
                       cfg=cfg)
