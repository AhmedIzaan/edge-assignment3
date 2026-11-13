"""Static PTQ: fusion -> observers -> training-only calibration -> INT8 conversion."""
from common import *
from torch.ao.quantization import prepare, convert, get_default_qconfig

def calibrate(model,loader):
    model.eval(); seen=0
    with torch.inference_mode():
        for x,_ in loader: model(x); seen+=len(x)
    return seen

def run(config,loaders=None):
    setup(config['seed'],config['threads'])
    _,_,calibration,_=loaders or data(config)
    model=load_float(); model.fuse_model(is_qat=False)
    model.qconfig=get_default_qconfig('fbgemm')
    prepare(model,inplace=True)
    seen=calibrate(model,calibration)
    print('PTQ calibration images:',seen,flush=True)
    model=convert(model.eval(),inplace=False)
    deploy(model,'ptq_int8')
    return model

if __name__=='__main__': run(load_config())
