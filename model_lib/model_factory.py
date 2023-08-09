import torch.nn as nn
import torch
from model_lib import timm
import importlib
from torchsummary import summary as summary

def create_model(model_name='', num_classes=0, pretrained=True, **kwargs):
    #Define model
    if str(type(pretrained)) != "<class 'bool'>":
        is_pretrained = False
    else:
        is_pretrained = pretrained

    if model_name.lower() == 'cafenet':
        model_def = importlib.import_module('model_lib.CaFeNet')  # dynamic import
        model = model_def.cafe_efficientnet(num_classes=num_classes, pretrained=is_pretrained)
    else:
        print(f'{model_name} is not implemented')
        return None

    print(f"Model LOADING...")
    #Load your own pretrained weight
    if str(type(pretrained)) != "<class 'bool'>":
        print(f"{pretrained.split('/')[-2]}/{pretrained.split('/')[-1]} model is LOADING!!")
        state_dict = torch.load(pretrained)
        replace_key = '' if model_name.lower() in ['cafenet', 'efficientnetb0', 'triplet', 'supcon'] else 'model.'

        for key in list(state_dict.keys()):
            tmp = key.replace('module.', replace_key)
            new_key = tmp.replace('model.model.', 'model.')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        model.load_state_dict(state_dict, strict=True)

    return model

