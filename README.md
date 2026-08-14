# cuda_games

Practice ground for the PyTorch / GPU track.

## Layout

| Path | What it is |
|---|---|
| `Tensor Puzzlers.ipynb` | Sasha Rush's Tensor-Puzzles. One line per puzzle, no `view` / `sum` / `take` / `squeeze` / `tensor`. |
| `GPU_puzzlers.ipynb` | Sasha Rush's GPU-Puzzles, CUDA thread indexing and shared memory. |
| `lib.py` | `make_test`, `run_test`, `draw_examples` used by both notebooks. |
| `benchmarking/bench.py` | Measurement harness: three timing routes compared, warm-up, repetition, peak memory, CSV with the run configuration. |
| `benchmarking/results.csv` | Rows produced by `bench.py`. Each row carries device, torch/cuda version, dtype, power state and the warm-up / number / repeat settings. |

## Setup

Python 3.14, torch 2.10 + cu128, RTX 4060 Laptop.

```
pip install torchtyping hypothesis pytest matplotlib colour chalk-diagrams
```

`lib.py` is vendored from [srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) so the notebooks run without the `wget` cell.

## Notes

Written up in the vault under `Skills/ML/PyTorch/` and `Skills/Hardware/GPU/`:
`GPU-Benchmarking`, `Host-Device-Synchronization`, `Tensor-Strides`, `CUDA-Streams`.
