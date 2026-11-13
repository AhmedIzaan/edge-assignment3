"""ONNX Runtime CPU inference with the same preprocessing as native deployment."""
import argparse
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnxruntime as ort
ort.disable_telemetry_events()  # Disable optional runtime telemetry before sessions are created.
from PIL import Image
from common import *

def predict(name,image=None,index=0):
    config=load_config()
    tf=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    if image: x=tf(Image.open(image).convert('RGB')).unsqueeze(0).numpy()
    else:
        ds=datasets.CIFAR10(config['data_dir'],train=False,download=True,transform=tf)
        x=ds[index][0].unsqueeze(0).numpy()
    opts=ort.SessionOptions(); opts.intra_op_num_threads=config['threads'];opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/'onnx'/f'{name}.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    logits=session.run(['logits'],{'images':x})[0]
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    i=int(p[0].argmax());return dict(model=name,class_index=i,class_name=CLASSES[i],softmax_score=float(p[0,i]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=NAMES,default='fp32')
    parser.add_argument('--image');parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args(); print(json.dumps(predict(args.model,args.image,args.index),indent=2))
