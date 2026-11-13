"""Export each deployed model to ONNX; validate structure and CPU numerical agreement."""
import onnx
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnxruntime as ort
ort.disable_telemetry_events()  # Disable optional runtime telemetry before sessions are created.
from common import *

def run(config,loader=None):
    setup(config['seed'],config['threads'])
    if loader is None: loader=data(config)[3]
    x=next(iter(loader))[0][:32]; rows=[]
    for name in NAMES:
        model=torch.jit.load(str(ROOT/'models'/f'{name}.pt'),map_location='cpu').eval()
        path=ROOT/'onnx'/f'{name}.onnx'
        torch.onnx.export(model,x[:1],str(path),opset_version=17,
            input_names=['images'],output_names=['logits'],
            dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},do_constant_folding=True)
        graph=onnx.load(str(path)); onnx.checker.check_model(graph)
        opts=ort.SessionOptions(); opts.intra_op_num_threads=config['threads']; opts.inter_op_num_threads=1
        sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        with torch.inference_mode(): expected=model(x).numpy()
        actual=sess.run(['logits'],{'images':x.numpy()})[0]
        error=float(np.max(np.abs(expected-actual)))
        agreement=float(np.mean(expected.argmax(1)==actual.argmax(1))*100)
        # Backend rounding can accumulate across layers; audit in final output steps.
        # Also measure full-test deployment accuracy and agreement below.
        output_step=float(model.fc.scale) if 'int8' in name else 0.0
        tolerance=3*output_step+1e-6 if 'int8' in name else 1e-3
        assert error<tolerance,(name,error)
        qnodes=sum(n.op_type=='QuantizeLinear' for n in graph.graph.node)
        if 'int8' in name: assert qnodes>0
        native_correct=onnx_correct=agree=total=0
        for batch,labels in loader:
            actual_logits=sess.run(['logits'],{'images':batch.numpy()})[0]
            with torch.inference_mode(): native_logits=model(batch).numpy()
            actual_pred=actual_logits.argmax(1); native_pred=native_logits.argmax(1)
            onnx_correct+=int((actual_pred==labels.numpy()).sum())
            native_correct+=int((native_pred==labels.numpy()).sum())
            agree+=int((actual_pred==native_pred).sum()); total+=len(labels)
        rows.append(dict(model=name,onnx_mib=path.stat().st_size/2**20,
            checked_images=len(x),max_absolute_logit_error=error,prediction_agreement_pct=agreement,
            quantize_nodes=qnodes,output_quantization_step=output_step,
            max_error_in_output_steps=error/output_step if output_step else 0.0,
            full_test_images=total,onnx_test_accuracy_pct=100*onnx_correct/total,
            native_test_accuracy_pct=100*native_correct/total,
            full_test_prediction_agreement_pct=100*agree/total,
            status='checker and CPU execution passed'))
        print(rows[-1],flush=True)
    df=pd.DataFrame(rows); df.to_csv(ROOT/'results'/'onnx_verification.csv',index=False)
    return df

if __name__=='__main__': run(load_config())
