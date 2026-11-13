# Assignment 3: ResNet-18 quantization and pruning

Student: Ahmed Izaan | Section: A

## Files
- Assignment_3_Solved.ipynb: complete implementation, explanations and executed outputs.
- scripts/: separate baseline, PTQ, calibration, QAT, pruning, inference, benchmarking and ONNX scripts.
- models/: retrainable FP32 states (.pth), five executable TorchScript models (.pt), and compressed unstructured state (.pth.gz).
- onnx/: five deployable ONNX models (INT8 models use Q/DQ graphs).
- results/: full-test summary/predictions CSVs, split indices, training logs, export checks and result figures.
- TECHNICAL_REPORT.md: implementation details, measured results and figures.

## Installation
Use Python 3.12 on x86 CPU/Linux (tested). From the extracted folder:

    python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt

Start Jupyter and open Assignment_3_Solved.ipynb. Run All recreates scripts and results.
Alternatively:

    python scripts/run_all.py

Before running scripts on another computer, set data_dir in config.json to a local writable folder, for example ./data. No CIFAR-10 data is included. The first run downloads it; allow enough disk space and network access.
The notebook sets a default data folder relative to the extracted project. Change config in its setup cell if desired.

## Individual stages
    python scripts/train_baseline.py
    python scripts/ptq.py
    python scripts/qat.py
    python scripts/prune_unstructured.py
    python scripts/prune_structured.py
    python scripts/benchmark.py
    python scripts/export_onnx.py

Run baseline training before optimization. Benchmark/export require all five checkpoints.

## CPU inference
    python scripts/inference.py --model qat_int8 --index 0
    python scripts/inference.py --model structured --image path/to/image.png
    python scripts/onnx_inference.py --model ptq_int8 --index 0

Model choices: fp32, ptq_int8, qat_int8, unstructured, structured.

## Reproduction and interpretation
Fixed seed 42, deterministic PyTorch operations where supported (warn-only for quantized kernels), four CPU threads, no test-set tuning. The run uses 45,000 training examples, 5,000 validation examples and all 10,000 test examples. Calibration uses 1,000 balanced images from training. Four baseline epochs and short recovery/QAT runs are recorded in config.json; increase baseline_epochs for stronger accuracy. Hardware and software affect latency and exact training results.

For the required classroom upload, rename the ZIP to YOUR-ROLL-NUM_A.zip. Roll number was not supplied and is not invented.
