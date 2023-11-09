

import random
import sys
import numpy as np
import torch
from dataset_loader import dataset_loader, data_loader
from loss import loss_fn
from models import get_model
from optim import optimizer

from rppg.config import get_config
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

    datasets = dataset_loader(fit_cfg=cfg.fit, dataset_path=cfg.dataset_path)
    data_loaders = data_loader(fit_cfg=cfg.fit, datasets=datasets)

    model = get_model(fit_cfg=cfg.fit)

    opt = None
    criterion = None
    lr_sch = None

    if cfg.fit.train.flag:
        opt = optimizer(model_params=model.parameters(), 
                        leanring_rate=cfg.fit.train.learning_rate, 
                        optim=cfg.fit.train.optimizer)
        criterion = loss_fn(loss_name=cfg.fit.train.loss)
        lr_sch = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=cfg.fit.train.learning_rate,
            steps_per_epoch=len(datasets[0]),
            epochs=cfg.fit.train.epochs
        )

    test_result = run(model=model, 
                      optimizer=opt, 
                      lr_sch=lr_sch, 
                      criterion=criterion, 
                      cfg=cfg, 
                      dataloaders=data_loaders)
    
    save_single_result(result_path=result_save_path, 
                       result=test_result, 
                       cfg=cfg)
    sys.exit(0)
    
    
