"""Iterative global magnitude pruning with masks preserved until fine-tuning ends."""
import gzip, shutil
from torch.nn.utils import prune
from common import *

def weight_modules(model):
    return [(m,'weight') for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]

def sparsity(model):
    modules=weight_modules(model)
    zero=sum((m.weight.detach()==0).sum().item() for m,_ in modules)
    total=sum(m.weight.numel() for m,_ in modules)
    return zero,total,100*zero/total

def run(config,loaders=None):
    setup(config['seed'],config['threads'])
    train,val,_,_=loaders or data(config)
    model=load_float(); params=weight_modules(model); rows=[]
    # A deterministic training-only recovery subset; no test examples.
    short=DataLoader(Subset(train.dataset,list(range(5000))),batch_size=config['batch_size'],shuffle=True)
    for iteration in range(config['pruning_iterations']):
        prune.global_unstructured(params,pruning_method=prune.L1Unstructured,
                                  amount=config['unstructured_amount_per_iteration'])
        z,n,s=sparsity(model)
        print(f'Iteration {iteration+1}: {z:,}/{n:,} weights zero ({s:.3f}%)',flush=True)
        fit(model,short,val,1,0.002,f'unstructured_iteration_{iteration+1}',keep_best=False)
        rows.append(dict(iteration=iteration+1,zero_weights=z,total_weights=n,sparsity_pct=s,
                         val_accuracy_pct=evaluate(model,val)['accuracy_pct']))
    fit(model,train,val,config['finetune_epochs'],0.001,'unstructured_final',keep_best=False)
    for module,name in params: prune.remove(module,name)
    assert not any('weight_mask' in key or 'weight_orig' in key for key in model.state_dict())
    assert abs(sparsity(model)[2]-100*(1-0.8**config['pruning_iterations']))<0.01
    pd.DataFrame(rows).to_csv(ROOT/'results'/'pruning_iterations.csv',index=False)
    save_float(model,'unstructured')
    src=ROOT/'models'/'unstructured.pth'
    with src.open('rb') as f, gzip.open(str(src)+'.gz','wb',compresslevel=9) as out:
        shutil.copyfileobj(f,out)
    print('Permanent masks removed; remaining zeros:',sparsity(model),flush=True)
    return model

if __name__=='__main__': run(load_config())
