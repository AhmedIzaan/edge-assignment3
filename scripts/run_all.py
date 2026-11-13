"""One-command reproduction of all training, optimization and deployment experiments."""
from common import *
from train_baseline import run as baseline
from ptq import run as ptq
from qat import run as qat
from prune_unstructured import run as unstructured
from prune_structured import run as structured
from benchmark import run as benchmark
from export_onnx import run as onnx_export

if __name__=='__main__':
    cfg=load_config();setup(cfg['seed'],cfg['threads']);loaders=data(cfg)
    baseline(cfg,loaders);ptq(cfg,loaders);qat(cfg,loaders)
    unstructured(cfg,loaders);structured(cfg,loaders)
    benchmark(cfg,loaders[3]);onnx_export(cfg,loaders[3])
