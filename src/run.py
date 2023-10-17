

import os
import numpy as np
import torch
from tqdm import tqdm

from utils.funcs import get_hr, MAE, RMSE, MAPE, corr


def run(model, optimizer, lr_sch, criterion, cfg, dataloaders):
    log = True
    best_loss = 100000
    val_loss = 0
    eval_flag = False
    save_dir = cfg.model_save_path + cfg.fit.model + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_result = []
    if cfg.fit.train_flag:
        for epoch in range(cfg.fit.train.epochs):
            train_fn(epoch, model, optimizer, lr_sch, criterion, dataloaders[0])
            val_loss = val_fn(epoch, model, criterion, dataloaders[1])
            if best_loss > val_loss:
                best_loss = val_loss
                eval_flag = True
                if cfg.fit.model_save_flag:
                    torch.save(model.state_dict(),
                        f'{save_dir}train{cfg.fit.train.dataset}_test{cfg.fit.test.dataset}_imagesize{str(cfg.fit.img_size)}.pt'
                    )
            if log:
                et = cfg.fit.test.eval_time_length
                if cfg.fit.eval_flag and (eval_flag or (epoch + 1) % cfg.fit.evel_interval == 0):
                    test_fn(model, dataloaders[2], cfg.fit.test.vital_type, cfg.fit.test.cal_type, cfg.fit.test.bpf, cfg.fit.test.metrics, et)
                    eval_flag = False

        test_result = test_fn(model, dataloaders[2], cfg.fit.test.vital_type, cfg.fit.test.cal_type, 
                              cfg.fit.test.bpf, cfg.fit.test.metrics, cfg.fit.test.eval_time_length)
    else:
        test_result.append(test_fn(model, dataloaders[0], cfg.fit.test.vital_type, cfg.fit.test.cal_type, 
                                   cfg.fit.test.bpf, cfg.fit.test.metrics, cfg.fit.test.eval_time_length))
    return test_result

        


def train_fn(epoch, model, optimizer, lr_sch, criterion, dataloader):
    step = 'Train'

    with tqdm(dataloader, desc=step, total=len(dataloader)) as tepoch:
        model.train()
        running_loss = 0.0

        for te in tepoch:
            inputs, target = te
            optimizer.zero_grad()
            tepoch.set_description(f"{step} Epoch {epoch}")
            outputs = model(inputs)
            loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            loss.requires_grad = True
            loss.backward(retain_graph=True)
            running_loss += loss.item()

            optimizer.step()
            if lr_sch is not None:
                lr_sch.step()
            
            tepoch.set_postfix({'': 'loss : %0.4f | ' % (running_loss / len(tepoch))})


def val_fn(epoch, model, criterion, dataloader):
    step = 'Val'

    with tqdm(dataloader, desc=step, total=len(dataloader)) as tepoch:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for te in tepoch:
                inputs, target = te
                tepoch.set_description(f"{step} Epoch {epoch}")
                outputs = model(inputs)
                loss = criterion(outputs, target)

                if ~torch.isfinite(loss):
                    continue

                running_loss += loss.item()
                tepoch.set_postfix({'': 'loss : %0.4f | ' % (running_loss / len(tepoch))})
        return running_loss / len(tepoch)
    

def test_fn(model, dataloaders, vital_type, cal_type, bpf, metrics, eval_time_length):
    step = "Test"
    model.eval()
    fs = 30

    interval = fs * eval_time_length
    empty_tensor = torch.empty(1).cuda()
    empty_tensor2 = torch.empty(1).cuda()
    with tqdm(dataloaders, desc=step, total=len(dataloaders), disable=False) as tepoch:
        with torch.no_grad():
            for te in tepoch:
                torch.cuda.empty_cache()
                inputs, target = te
                outputs = model(inputs)
                empty_tensor = torch.cat(empty_tensor, outputs.squeeze(), dim=-1)
                empty_tensor2 = torch.cat(empty_tensor2, target.squeeze(), dim=-1)
    pred_chunks = torch.stack(list(torch.split(empty_tensor[1:].detach(), interval))[:-1], dim=0)
    target_chunks = torch.stack(list(torch.split(empty_tensor2[1:].detach(), interval))[:-1], dim=0)

    hr_pred, hr_target = get_hr(pred_chunks, target_chunks, vital_type=vital_type, cal_type=cal_type, fs=fs, bpf=bpf)

    hr_pred = np.asarray(hr_pred.detach().cpu())
    hr_target = np.asarray(hr_target.detach().cpu())

    test_result = []
    if 'MAE' in metrics:
        test_result.append(round(MAE(hr_pred, hr_target), 3))
        print('MAE : ', MAE(hr_pred, hr_target))
    if 'RMSE' in metrics:
        test_result.append(round(RMSE(hr_pred, hr_target), 3))
        print('RMSE : ', np.sqrt(np.mean((hr_pred - hr_target) ** 2)))
    if 'MAPE' in metrics:
        test_result.append(round(MAPE(hr_pred, hr_target), 3))
        print('MAPE : ', MAPE(hr_pred, hr_target))
    if 'Pearson' in metrics:
        test_result.append(round(corr(hr_pred, hr_target)[0][1], 3))
        print('Pearson : ', corr(hr_pred, hr_target))
    return test_result


