
from rppg.nets.DeepPhys import DeepPhys


def get_model(fit_cfg):
    model_name = fit_cfg.model
    time_length = fit_cfg.time_length
    img_size = fit_cfg.img_size

    if model_name == 'DeepPhys':
        model = DeepPhys()
    else:
        raise NotImplementedError('Model not implemented (%s)' % model_name)
    
    return model.cuda()