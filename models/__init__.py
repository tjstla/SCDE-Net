from .SCDENet import *

def get_model(name, net=None):
    if name.lower() in ['scdenet', 'scde-net', 'scde']:
        net = SCDENet(stage_num=6)
    else:
        raise NotImplementedError(f"Unknown model name: {name}. Supported: 'scdenet'.")

    return net
