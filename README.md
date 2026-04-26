# CUDA Batch Edge Detector

CUDA Batch Edge Detector is a CUDA C++ image-processing project for the
Coursera CUDA at Scale independent project. It generates hundreds of small
synthetic grayscale images, processes them on the GPU, and writes edge maps plus
a timing/quality summary that can be committed as proof of execution.

## What It Does

The program reads a directory of binary PGM (`P5`) grayscale images and applies
two CUDA kernels to every image in one CLI run:

1. A 3x3 Gaussian blur kernel reduces synthetic sensor noise.
2. A Sobel edge-detection kernel computes gradient magnitude and thresholds the
   result into a black/white edge map.

This is not a CPU-only implementation. The image transforms are performed by
CUDA kernels in `src/main.cu`; the CPU only handles file I/O, argument parsing,
and summary reporting.

## Repository Contents

- `src/main.cu` - CUDA C++ CLI, PGM I/O, GPU kernels, and batch driver.
- `scripts/generate_dataset.py` - deterministic generator for 256 grayscale
  test images.
- `scripts/run_demo.sh` - Linux/Coursera-lab end-to-end build and run script.
- `scripts/run_demo.ps1` - Windows PowerShell variant.
- `CMakeLists.txt` - CMake build for CUDA.
- `Makefile` - convenience targets for configure, build, demo, and clean.
- `proof/` - place to store logs, summaries, and selected output images from a
  real CUDA run.

## Build Requirements

- NVIDIA GPU with CUDA-capable driver.
- CUDA Toolkit with `nvcc`.
- CMake 3.18 or newer.
- Python 3 for dataset generation.

## Quick Start

On a Linux CUDA lab machine:

```bash
make demo
```

That command generates 256 input images, builds the CUDA executable, processes
all generated images, and stores proof artifacts under `proof/latest/`.

Equivalent explicit commands:

```bash
python3 scripts/generate_dataset.py --output data/input --count 256 --width 256 --height 256 --seed 7
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda_batch_edges --input data/input --output data/output --threshold 88
```

If you want to run the script directly, use `bash scripts/run_demo.sh`.

On Windows PowerShell:

```powershell
.\scripts\run_demo.ps1
```

## CLI

```text
cuda_batch_edges --input <input_dir> --output <output_dir> [options]

Options:
  --threshold <0-255>   Edge threshold. Default: 88
  --max-images <n>      Process at most n images. Default: 0, meaning all
  --warmup <n>          Warmup GPU launches before timed work. Default: 1
  --help                Show usage
```

Example:

```bash
./build/cuda_batch_edges --input data/input --output data/output --threshold 96 --max-images 128
```

## Proof of Execution

After running `make demo`, commit the files in `proof/latest/` to your public
repository. The important proof files are:

- `run.log` - includes GPU device name, image count, total pixels, and timing.
- `summary.csv` - one row per processed image.
- `outputs/` - selected generated edge-map images.
- `contact_sheet.png` - browser-friendly preview of selected edge maps.

The log should show a single invocation processing hundreds of files, for
example `Processed images: 256`.

## Short Project Description

This project performs batch image processing on a CUDA GPU. I chose edge
detection because it is easy to inspect visually but still exercises a real
image pipeline: memory transfer, neighborhood filtering, gradient computation,
thresholding, and per-image reporting. The project processes hundreds of small
synthetic images generated with deterministic geometric patterns and noise. The
GPU implementation uses two custom CUDA kernels, one for Gaussian blur and one
for Sobel edge detection. The biggest practical challenge was keeping the
project portable for peer reviewers, so the code uses plain PGM files and avoids
external image-processing dependencies.
