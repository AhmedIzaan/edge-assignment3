"""Shared architecture, deterministic CIFAR-10 data, training and deployment utilities."""
import copy, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torch.ao.quantization import QuantStub, DeQuantStub

ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
NAMES = ['fp32', 'ptq_int8', 'qat_int8', 'unstructured', 'structured']

def setup(seed=42, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if 'fbgemm' not in torch.backends.quantized.supported_engines:
        raise RuntimeError('This experiment requires an x86 CPU build with FBGEMM.')
    torch.backends.quantized.engine = 'fbgemm'
    for d in ('models','results','onnx'):
        (ROOT / d).mkdir(exist_ok=True)

def architecture(hidden_channels=None):
    # Keep the original 7x7/stride-2 stem and max pool: explicit CPU cost tradeoff.
    model = resnet18(weights=None, num_classes=10, quantize=False)
    if hidden_channels:
        for name, width in hidden_channels.items():
            block = model.get_submodule(name)
            old1, old2 = block.conv1, block.conv2
            block.conv1 = nn.Conv2d(old1.in_channels, width, 3,
                                    old1.stride, old1.padding, bias=False)
            block.bn1 = nn.BatchNorm2d(width)
            block.conv2 = nn.Conv2d(width, old2.out_channels, 3,
                                    old2.stride, old2.padding, bias=False)
    return model

def data(config):
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    eval_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    folder = config['data_dir']
    train_aug = datasets.CIFAR10(folder,train=True,download=True,transform=train_tf)
    train_eval = datasets.CIFAR10(folder,train=True,download=False,transform=eval_tf)
    test = datasets.CIFAR10(folder,train=False,download=True,transform=eval_tf)
    labels = np.asarray(train_aug.targets)
    rng = np.random.default_rng(config['seed'])
    train_ids, val_ids = [], []
    for c in range(10):
        ids = rng.permutation(np.where(labels == c)[0])
        val_ids.extend(ids[:500].tolist()); train_ids.extend(ids[500:].tolist())
    train_ids = rng.permutation(train_ids).tolist()
    val_ids = sorted(val_ids)
    # Balanced calibration subset: only training images, no augmented/random crop.
    calibration_ids = []
    for c in range(10):
        calibration_ids.extend([i for i in train_ids if labels[i] == c][:100])
    kw = dict(batch_size=config['batch_size'],num_workers=0)
    generator = torch.Generator().manual_seed(config['seed'])
    train = DataLoader(Subset(train_aug,train_ids),shuffle=True,generator=generator,**kw)
    val = DataLoader(Subset(train_eval,val_ids),shuffle=False,**kw)
    calibration = DataLoader(Subset(train_eval,calibration_ids),shuffle=False,**kw)
    test_loader = DataLoader(test,shuffle=False,**kw)
    np.savez(ROOT/'results'/'data_split.npz',train=np.array(train_ids),
             validation=np.array(val_ids),calibration=np.array(calibration_ids))
    return train,val,calibration,test_loader

@torch.inference_mode()
def evaluate(model, loader):
    model.eval(); correct = count = 0; loss = 0.0
    for x,y in loader:
        logits=model(x); loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(1)==y).sum().item(); count+=len(y)
    return {'loss':loss/count,'accuracy_pct':100*correct/count,'n':count}

def fit(model, train, val, epochs, lr, tag, qat=False, keep_best=True):
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(epochs,1))
    rows=[]; best=-1; best_state=None
    for epoch in range(epochs):
        start=time.perf_counter(); model.train()
        if qat:
            # Preserve pretrained BN statistics throughout the short QAT fine-tune.
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        total=correct=0; loss_sum=0
        for step,(x,y) in enumerate(train):
            if qat and epoch==epochs-1 and step==max(1,len(train)//2):
                model.apply(torch.ao.quantization.disable_observer)
            optimizer.zero_grad(set_to_none=True)
            out=model(x); loss=nn.functional.cross_entropy(out,y)
            loss.backward(); optimizer.step()
            total+=len(y); correct+=(out.argmax(1)==y).sum().item(); loss_sum+=loss.item()*len(y)
        metric=evaluate(model,val)
        row=dict(stage=tag,epoch=epoch+1,train_loss=loss_sum/total,
                 train_accuracy_pct=100*correct/total,val_accuracy_pct=metric['accuracy_pct'],
                 seconds=time.perf_counter()-start,learning_rate=optimizer.param_groups[0]['lr'])
        rows.append(row); print(row,flush=True)
        if metric['accuracy_pct']>best:
            best=metric['accuracy_pct']; best_state=copy.deepcopy(model.state_dict())
        scheduler.step()
    if keep_best and best_state is not None: model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'{tag}_training.csv',index=False)
    return model.eval()

def save_float(model,name,hidden_channels=None):
    torch.save({'state_dict':model.state_dict(),'hidden_channels':hidden_channels or {},
                'architecture':'torchvision quantizable ResNet-18, original stem, 10 classes'},
               ROOT/'models'/f'{name}.pth')
    deploy(model,name)

def load_float(name='fp32'):
    ckpt=torch.load(ROOT/'models'/f'{name}.pth',map_location='cpu',weights_only=True)
    m=architecture(ckpt['hidden_channels']); m.load_state_dict(ckpt['state_dict'])
    return m.eval()

def deploy(model,name):
    # TorchScript file is directly executable without reconstructing a Python class.
    model.eval(); scripted=torch.jit.script(model)
    scripted.save(str(ROOT/'models'/f'{name}.pt'))
    loaded=torch.jit.load(str(ROOT/'models'/f'{name}.pt')).eval()
    x=torch.randn(2,3,32,32)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x),model(x),rtol=1e-4,atol=1e-5)
    print(f'Saved and reloaded {name}: output shape {tuple(loaded(x).shape)}',flush=True)

def load_config():
    return json.loads((ROOT/'config.json').read_text())
