"""Reload all checkpoints; measure full-test accuracy, predictions, warm latency and throughput."""
from common import *

@torch.inference_mode()
def run(config,loader=None):
    setup(config['seed'],config['threads'])
    if loader is None: loader=data(config)[3]
    summary=[]; predictions=[]
    for name in NAMES:
        model=torch.jit.load(str(ROOT/'models'/f'{name}.pt'),map_location='cpu').eval()
        probe=torch.randn(1,3,32,32)
        for _ in range(20): model(probe)
        latency=[]
        for _ in range(100):
            start=time.perf_counter_ns(); model(probe)
            latency.append((time.perf_counter_ns()-start)/1e6)
        correct=total=0; inference_seconds=0.0
        for batch_index,(x,y) in enumerate(loader):
            start=time.perf_counter_ns(); logits=model(x)
            elapsed_ms=(time.perf_counter_ns()-start)/1e6
            inference_seconds+=elapsed_ms/1000
            probabilities=logits.softmax(1); confidence,pred=probabilities.max(1)
            for j in range(len(y)):
                predictions.append(dict(model=name,test_index=total+j,true_label=int(y[j]),
                    predicted_label=int(pred[j]),true_class=CLASSES[int(y[j])],predicted_class=CLASSES[int(pred[j])],
                    correct=bool(pred[j]==y[j]),softmax_score=float(confidence[j]),batch_index=batch_index,
                    batch_size=len(y),batch_inference_ms=elapsed_ms,amortized_ms_per_image=elapsed_ms/len(y)))
            correct+=(pred==y).sum().item(); total+=len(y)
        row=dict(model=name,test_images=total,accuracy_pct=100*correct/total,
                 checkpoint_mib=(ROOT/'models'/f'{name}.pt').stat().st_size/2**20,
                 latency_p50_ms=float(np.percentile(latency,50)),latency_p95_ms=float(np.percentile(latency,95)),
                 latency_p99_ms=float(np.percentile(latency,99)),throughput_images_s=total/inference_seconds,
                 inference_only_seconds=inference_seconds,threads=config['threads'])
        summary.append(row); print(row,flush=True)
    frame=pd.DataFrame(summary); pred_frame=pd.DataFrame(predictions)
    assert len(pred_frame)==5*len(loader.dataset)
    assert pred_frame.groupby('model').test_index.nunique().eq(len(loader.dataset)).all()
    frame.to_csv(ROOT/'results'/'benchmark.csv',index=False)
    pred_frame.to_csv(ROOT/'results'/'test_predictions.csv',index=False)
    frame.iloc[:3].to_csv(ROOT/'results'/'question1_benchmark.csv',index=False)
    frame[frame.model.isin(['fp32','unstructured','structured'])].to_csv(ROOT/'results'/'question2_benchmark.csv',index=False)
    return frame,pred_frame

if __name__=='__main__': run(load_config())
