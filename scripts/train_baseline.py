"""Train FP32 ResNet-18 on CIFAR-10 and export its checkpoints."""
from common import *

def run(config,loaders=None):
    setup(config['seed'],config['threads'])
    train,val,_,_=loaders or data(config)
    model=architecture()
    print('Trainable baseline parameters:',sum(p.numel() for p in model.parameters()),flush=True)
    fit(model,train,val,config['baseline_epochs'],0.05,'baseline')
    save_float(model,'fp32')
    return model

if __name__=='__main__': run(load_config())
