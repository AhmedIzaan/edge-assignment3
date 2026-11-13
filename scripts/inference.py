"""CPU inference for all five native deployment models."""
import argparse
from PIL import Image
from common import *

def predict(name,image=None,index=0):
    setup(); model=torch.jit.load(str(ROOT/'models'/f'{name}.pt'),map_location='cpu').eval()
    transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    if image: x=transform(Image.open(image).convert('RGB')).unsqueeze(0)
    else:
        ds=datasets.CIFAR10(load_config()['data_dir'],train=False,download=True,transform=transform)
        x=ds[index][0].unsqueeze(0)
    with torch.inference_mode(): score=model(x).softmax(1)[0]
    i=int(score.argmax()); return {'model':name,'class_index':i,'class_name':CLASSES[i],'softmax_score':float(score[i])}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=NAMES,default='fp32')
    parser.add_argument('--image'); parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args(); print(json.dumps(predict(args.model,args.image,args.index),indent=2))
