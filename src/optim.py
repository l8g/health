import torch.optim as opt

def optimizer(model_params, leanring_rate =1, optim = 'Adam'):

    if optim == 'Adam':
        return opt.Adam(model_params, lr=leanring_rate, weight_decay=5e-5)
    elif optim == 'SGD':
        return opt.SGD(model_params, lr=leanring_rate)
    elif optim == 'AdamW':
        return opt.AdamW(model_params, lr=leanring_rate)
    else:
        raise NotImplementedError('Optimizer not implemented (%s)' % optim)
    