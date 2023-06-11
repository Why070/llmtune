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
    for n, m in model.named_modules():
        if '4bit' in str(type(m)) or 'QuantLinear' in str(type(m)):
            print(f"Parameters in module '{n}':")
            print(f"Zeros: {m.zeros} (dtype: {m.zeros.dtype}), number of parameters: {m.zeros.numel()}")
            print(f"Scales: {m.scales} (dtype: {m.scales.dtype}), number of parameters: {m.scales.numel()}")

def download_file(url, path):
	print('Starting download')
	urllib.request.urlretrieve(url, path)
	print('Done')
