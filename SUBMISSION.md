# Peer-Graded Submission Notes

## Repository URL

After pushing this folder to GitHub/GitLab/Bitbucket, submit the public
repository URL for the assignment.

## Short Description

This project is a CUDA C++ batch image-processing program. It generates 256
synthetic grayscale PGM images, then processes the full image set in one CLI run
on the GPU. Each image is blurred with a custom 3x3 Gaussian CUDA kernel and
then passed through a custom Sobel threshold CUDA kernel to create an edge map.
The program writes processed output images and a `summary.csv` file containing
per-image dimensions, edge-pixel counts, edge ratios, and measured CUDA kernel
time. I used plain PGM files so reviewers can build and run the project without
extra image libraries, while still demonstrating real GPU computation over a
large batch of inputs.

## Proof of Execution

Run this on a CUDA-enabled lab machine:

```bash
make demo
```

Then commit:

```text
proof/latest/dataset.log
proof/latest/run.log
proof/latest/summary.csv
proof/latest/contact_sheet.png
proof/latest/outputs/*.pgm
```

The key line for reviewers in `proof/latest/run.log` is:

```text
Processed images: 256
```

The log also prints the detected NVIDIA GPU, compute capability, total pixels,
accumulated kernel time, and wall-clock time.

## Rubric Coverage

- Public code repository: push this folder to a public repo.
- README: `README.md` explains the project and how to run it.
- CLI with arguments: `cuda_batch_edges --input --output --threshold --max-images --warmup`.
- GPU computation: `src/main.cu` contains CUDA kernels and CUDA runtime calls.
- Compile/run support: `CMakeLists.txt`, `Makefile`, `scripts/run_demo.sh`, and
  `scripts/run_demo.ps1`.
- Proof artifacts: generated under `proof/latest/` by `make demo`.
