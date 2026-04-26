# Proof Artifacts

Run `make demo` on a CUDA-enabled machine, then commit the generated
`proof/latest/` directory.

Expected files:

- `dataset.log` - confirms the generated image count and total pixels.
- `run.log` - confirms CUDA device details and batch execution.
- `summary.csv` - one output row for each processed image.
- `outputs/*.pgm` - selected edge-map outputs for visual inspection.
- `contact_sheet.png` - browser-friendly preview of selected edge maps.

The peer grader should be able to see in `run.log` that one command processed
hundreds of images with GPU kernels.
