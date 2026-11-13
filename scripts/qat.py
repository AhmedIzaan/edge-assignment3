"""QAT from the pretrained baseline, followed by real INT8 conversion."""
from common import *
from torch.ao.quantization import prepare_qat,convert,get_default_qat_qconfig,FakeQuantizeBase

def run(config,loaders=None):
    setup(config['seed'],config['threads'])
    train,val,_,_=loaders or data(config)
    model=load_float().train(); model.fuse_model(is_qat=True)
    model.qconfig=get_default_qat_qconfig('fbgemm')
    prepare_qat(model,inplace=True)
    print('Inserted fake quantization modules:',sum(isinstance(m,FakeQuantizeBase) for m in model.modules()),flush=True)
    fit(model,train,val,config['qat_epochs'],0.001,'qat',qat=True,keep_best=False)
    model=convert(model.eval(),inplace=False)
    deploy(model,'qat_int8')
    return model

if __name__=='__main__': run(load_config())
