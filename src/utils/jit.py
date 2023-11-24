
import torch

import sys
sys.path.append('..')
from rppg.nets.DeepPhys import DeepPhys


def get_jit_model(model, input_shape, device, input_names=None, output_names=None):
    """
    Get the jit model from the given model.
    """
    model.eval()
    model.to(device)
    example_input = torch.rand(input_shape).to(device), torch.rand(input_shape).to(device)
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save('model.pt')
    return traced_script_module


def get_jit_model_from_path(model_path, input_shape, device, input_names=None, output_names=None):
    """
    Get the jit model from the given model path.
    """
    model_dict = torch.load(model_path)
    model = DeepPhys()
    model.load_state_dict(model_dict)
    return get_jit_model(model, input_shape, device, input_names, output_names)

if __name__ == '__main__':
    model_path = './../models/best_model.pth'
    input_shape = (180, 3, 128, 128)
    device = torch.device('cpu')
    model = get_jit_model_from_path(model_path, input_shape, device)
    print(model)