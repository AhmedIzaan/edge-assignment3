"""Physical dependency-aware channel removal inside all eight ResNet-18 BasicBlocks."""
from common import *

def reconstruct(original,fraction):
    model=copy.deepcopy(original).eval(); hidden={}; rows=[]
    for name,block in list(model.named_modules()):
        if not (name.startswith('layer') and name.count('.')==1): continue
        conv1,bn1,conv2=block.conv1,block.bn1,block.conv2
        score=conv1.weight.detach().abs().mean(dim=(1,2,3))
        keep_count=max(1,round(conv1.out_channels*(1-fraction)))
        keep=score.topk(keep_count).indices.sort().values
        new1=nn.Conv2d(conv1.in_channels,keep_count,3,conv1.stride,conv1.padding,bias=False)
        new_bn=nn.BatchNorm2d(keep_count,eps=bn1.eps,momentum=bn1.momentum)
        new2=nn.Conv2d(keep_count,conv2.out_channels,3,conv2.stride,conv2.padding,bias=False)
        with torch.no_grad():
            new1.weight.copy_(conv1.weight[keep])
            new2.weight.copy_(conv2.weight[:,keep])
            for attr in ('weight','bias','running_mean','running_var'):
                getattr(new_bn,attr).copy_(getattr(bn1,attr)[keep])
            new_bn.num_batches_tracked.copy_(bn1.num_batches_tracked)
        block.conv1,block.bn1,block.conv2=new1,new_bn,new2
        hidden[name]=keep_count
        rows.append(dict(block=name,original_hidden=conv1.out_channels,kept_hidden=keep_count,
                         removed_hidden=conv1.out_channels-keep_count))
    model.eval()
    with torch.inference_mode(): assert model(torch.randn(2,3,32,32)).shape==(2,10)
    assert all('mask' not in k for k in model.state_dict())
    return model,hidden,rows

def run(config,loaders=None):
    setup(config['seed'],config['threads'])
    train,val,_,_=loaders or data(config)
    original=load_float(); model,hidden,rows=reconstruct(original,config['channel_prune_fraction'])
    before=sum(p.numel() for p in original.parameters()); after=sum(p.numel() for p in model.parameters())
    assert after<before
    print(f'Physical parameters: {before:,} -> {after:,} ({100*(1-after/before):.2f}% fewer)',flush=True)
    pre=evaluate(model,val)['accuracy_pct']
    fit(model,train,val,config['finetune_epochs'],0.002,'structured',keep_best=True)
    post=evaluate(model,val)['accuracy_pct']
    print(f'Structured validation accuracy before/after fine-tuning: {pre:.2f}% / {post:.2f}%',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results'/'structured_channels.csv',index=False)
    (ROOT/'results'/'structured_recovery.json').write_text(json.dumps({'before_val_accuracy_pct':pre,
        'after_val_accuracy_pct':post,'parameters_before':before,'parameters_after':after},indent=2))
    save_float(model,'structured',hidden)
    # Verify architecture reconstruction from checkpoint metadata, not only TorchScript.
    restored=load_float('structured')
    with torch.inference_mode():
        x=torch.randn(2,3,32,32); torch.testing.assert_close(restored(x),model(x))
    return model

if __name__=='__main__': run(load_config())
