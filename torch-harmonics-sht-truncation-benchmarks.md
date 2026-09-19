# SHT truncation benchmark record

Date: 2026-09-19

This is a local benchmark record for the pull-request discussion. It is not a
repository artifact.

## Environment

- Host CPU: AMD Ryzen 9 5950X 16-Core Processor, 32 logical CPUs; the
  Legendre comparison used 16 PyTorch CPU threads.
- GPU: NVIDIA GeForce RTX 5070, driver 610.57.04, compute capability 12.0.
- Python: 3.13.13 (conda-forge).
- PyTorch: 2.14.0+cu130; `torch.version.cuda` 13.0.
- Platform: Linux 7.0.14-15-pve, glibc 2.43.

## Legendre comparison

The comparison was run in separate temporary worktrees with the same inline
benchmark: 3 warmup calls and 10 timed calls, median reported, 16 CPU
threads, CUDA synchronization around GPU calls. The cases were:

- `triangular_43`: `lmax=43`, `mmax=43`, `nlat=64`.
- `trapezoidal_43x85`: `lmax=85`, `mmax=43`, `nlat=128`.
- `triangular_128`: `lmax=128`, `mmax=128`, `nlat=180`.
- `rhomboidal_bbox_128x255`: `lmax=255`, `mmax=128`, `N=127`,
  `nlat=360`; this is the restricted recurrence/precompute path.

Commands/configurations:

- baseline worktree: `/tmp/torch-harmonics-bench-baseline` at
  `0bfa0ff2c3a65500985cc5f0223e66492db0777d`.
- optimized worktree: `/tmp/torch-harmonics-bench-optimized` at
  `5ddbc36c22207e3dee3a5ff8510e62d97d2c956f`.
- executable: `/home/albert/miniforge3/bin/python`.
- each case timed `legpoly` and `dlegpoly`; the restricted case supplied
  `l_minus_m_max=127` to the low-level implementation.

Times are milliseconds; the percentage is `(optimized - baseline) / baseline`.
Positive percentages are slower.

| device | case / function | baseline | 5ddbc36 | change |
|---|---|---:|---:|---:|
| CPU | triangular_43 / legpoly | 3.789 | 3.621 | -4.43% |
| CPU | triangular_43 / dlegpoly | 5.602 | 5.361 | -4.30% |
| CPU | trapezoidal_43x85 / legpoly | 8.578 | 11.676 | +36.12% |
| CPU | trapezoidal_43x85 / dlegpoly | 12.636 | 12.971 | +2.65% |
| CPU | triangular_128 / legpoly | 15.850 | 15.703 | -0.93% |
| CPU | triangular_128 / dlegpoly | 67.193 | 68.625 | +2.13% |
| CPU | rhomboidal_bbox_128x255 / legpoly | 44.335 | 46.297 | +4.43% |
| CPU | rhomboidal_bbox_128x255 / dlegpoly | 243.725 | 246.034 | +0.95% |
| CUDA | triangular_43 / legpoly | 10.162 | 10.358 | +1.93% |
| CUDA | triangular_43 / dlegpoly | 11.359 | 11.271 | -0.77% |
| CUDA | trapezoidal_43x85 / legpoly | 20.785 | 21.292 | +2.44% |
| CUDA | trapezoidal_43x85 / dlegpoly | 21.548 | 22.163 | +2.85% |
| CUDA | triangular_128 / legpoly | 34.549 | 35.359 | +2.34% |
| CUDA | triangular_128 / dlegpoly | 36.203 | 33.534 | -7.37% |
| CUDA | rhomboidal_bbox_128x255 / legpoly | 67.087 | 62.170 | -7.33% |
| CUDA | rhomboidal_bbox_128x255 / dlegpoly | 68.695 | 68.143 | -0.80% |

These fresh isolated reruns are noisy, especially for the small CPU cases.
The earlier raw 10-sample logs are retained at `/tmp/legendre-before.txt`
and `/tmp/legendre-after.txt`; they show the same environment and additional
run-to-run variation. The optimization from `5ddbc36` is retained unchanged;
these measurements are evidence, not a claim that every individual case is
faster on every run.

## SHT runtime regression

Command:

```text
/home/albert/miniforge3/bin/python benchmarks/run.py --name sht
```

This completed successfully on the current worktree (`5ddbc36` plus the
uncommitted finalization changes). The command reported:

| benchmark | device | fwd ms | bwd ms |
|---|---|---:|---:|
| `sht_fwd_bwd_1deg_b8_float32_cpu` | CPU | 0.67 | 2.88 |
| `sht_fwd_bwd_1deg_b4096_float32_cuda` | CUDA | 0.30 | 0.45 |
| `sht_fwd_bwd_hdeg_b1_float32_cuda` | CUDA | 0.31 | 0.45 |
| `isht_fwd_bwd_1deg_b8_float32_cpu` | CPU | 0.68 | 2.71 |
| `isht_fwd_bwd_1deg_b4096_float32_cuda` | CUDA | 0.39 | 0.48 |
| `isht_fwd_bwd_hdeg_b1_float32_cuda` | CUDA | 0.38 | 0.46 |

The run emitted only the existing default-truncation warning for the CPU
forward case. Earlier raw before/after CSVs from this work remain at
`/tmp/sht-before.csv`, `/tmp/sht-after-final.csv`, and their repeat files;
their small differences are within the observed run-to-run variability.
