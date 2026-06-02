# DC-FNCD — Usage Guide

A divide-and-conquer federated causal discovery toolkit with GPU-accelerated
conditional independence (CI) testing. It runs on CuPy or PyTorch when a GPU is
available and falls back to a pure NumPy implementation otherwise.

## Files

| File | Role |
| --- | --- |
| `dc_fncd_speed.py` | High-level entry point. Exposes the `DC-FNCD` algorithm and the speed benchmarks. |
| `federated_cf_ci_test_gpu.py` | Low-level engine. Implements the GPU/CPU conditional independence test used by the algorithm. |

## Requirements

- Python 3.10+ (developed and tested on 3.12)
- `numpy` (required)
- One acceleration backend, optional but recommended:
  - `cupy` — for NVIDIA GPUs (CUDA)
  - `torch` — PyTorch, CUDA or CPU

If neither `cupy` nor `torch` is installed, the code automatically falls back to
the NumPy backend.

```bash
pip install numpy
# pick ONE accelerator (optional):
pip install cupy-cuda12x      # match your CUDA version
# or
pip install torch
```

## Directory Layout

The high-level file imports the engine via
`from Code_DC_FNCD.federated_cf_ci_test_gpu import ...`, so both files must live
inside a package directory named `Code_DC_FNCD`:

```
Code_DC_FNCD/
├── __init__.py                  # can be empty
├── dc_fncd_speed.py
└── federated_cf_ci_test_gpu.py
```

If the import fails, the module degrades gracefully to the pure NumPy path.

## Quick Start

```python
from Code_DC_FNCD.dc_fncd_speed import dc_fncd

# data: per-client datasets, each an (n_samples, n_variables) array
result = dc_fncd(data)
```

`dc_fncd` runs the divide-and-conquer federated structure learning over the
provided client datasets and returns the recovered graph structure.

### Parameters

- `data` — the federated input: one dataset per client, each shaped
  `(n_samples, n_variables)`.
- Algorithm controls such as the CI-test significance level, conditioning-set
  size, and backend selection are passed through to the underlying CI test.

### Return Value

The learned causal structure (adjacency/skeleton plus any oriented edges),
suitable for downstream analysis or comparison against a ground-truth graph.

## Running the Benchmarks

`dc_fncd_speed.py` ships with built-in speed benchmarks across three problem
sizes — **Small**, **Medium**, and **Large** — that report timing for each
available backend (NumPy / CuPy / PyTorch).

```bash
python -m Code_DC_FNCD.dc_fncd_speed
```

## Using the CI Test Directly

You can call the low-level conditional independence test on its own:

```python
from Code_DC_FNCD.federated_cf_ci_test_gpu import federated_ci_test

# Test whether X ⊥ Y | Z across federated clients
independent, statistic = federated_ci_test(x, y, z, data)
```

This is useful for unit-testing edges or building a custom search procedure on
top of the federated CI primitive.

## Notes

- The backend is selected automatically: CuPy → PyTorch → NumPy, in order of
  availability.
- Logic is identical across backends; only the numerical kernels differ.
- All comments and docstrings have been removed and all source text is in
  English.
