import torch.nn as nn
import urllib.request

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def to_half_precision(model):
    for n, m in model.named_modules():
        if '4bit' in str(type(m)) or 'QuantLinear' in str(type(m)):
            m.zeros = m.zeros.half()
            m.scales = m.scales.half()    
    return model

def print_para(model):
    for name, module in model.named_modules():
        if '4bit' in str(type(module)) or 'QuantLinear' in str(type(module)):
            print(f"Parameters in module '{name}':")
            for param in module.parameters():
                print(param.shape, param.dtype)

def download_file(url, path):
	print('Starting download')
	urllib.request.urlretrieve(url, path)
	print('Done')
