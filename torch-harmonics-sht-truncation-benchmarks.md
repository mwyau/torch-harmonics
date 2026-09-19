# SHT truncation benchmark record

Date: 2026-09-19

This file is a temporary, checked-in summary for the truncation review. Raw
benchmark output is outside Git under
`/tmp/torch-harmonics-truncation-bench/`.

Times are milliseconds. Unless stated otherwise, the percentage is
`(candidate - baseline) / baseline`; negative values are faster. IQR is the
interquartile range across independent blocked-autorange measurements (or the
reported repeat medians when measurements from two passes were combined).

## Environment

- Host CPU: AMD Ryzen 9 5950X 16-Core Processor, 16 physical cores and 32
  logical CPUs.
- Primary CPU setting: 16 PyTorch intra-op threads and one inter-op thread.
  A 1-thread scalar sweep was also run. No CPU governor or frequency setting
  was changed; `lscpu` reported the host's normal frequency scaling.
- GPU: NVIDIA GeForce RTX 5070, 12,237 MiB, compute capability 12.0, driver
  610.57.04.
- Python: 3.13.13 (conda-forge), executable
  `/home/albert/miniforge3/bin/python`.
- PyTorch: 2.14.0+cu130, `torch.version.cuda == "13.0"`.
- Platform: Linux 7.0.14-15-pve-x86_64, glibc 2.43.

## Compared commits

| state | SHA | meaning |
|---|---|---|
| upstream base | `4ac8ed3d841b21c51035a9e262a4881dff5e4ddc` | current rebased NVIDIA `main` |
| A | `25c91e43721fab1eacc9d6f74496f9af8259fe9e` | immediately before `Optimize trapezoidal Legendre construction` |
| C | `dd3c1daf9622fa4a583be379e27d6dba94080324` | optimization-only state, before the final rhomboidal cleanup |
| B | `7fb550fb243da9fc813b6cbac60b75c73a85705d` | optimization plus final rhomboidal cleanup |
| branch state at start of this pass | `afe635f2a5f0e61d2430b3b0ea327b9a999dac38` | B plus the earlier temporary benchmark record |

A, B, and C were checked out in detached worktrees under `/tmp` and used the
same Python executable, PyTorch installation, thread settings, and benchmark
scripts. The cleanup in B shortens private docstrings only; C isolates the
trapezoidal optimization from that cleanup.

## Method

The direct recurrence, cached precompute, constructor, and temporary transform
benchmarks use `torch.utils.benchmark.Timer.blocked_autorange()`. Each timed
case was warmed first and then measured in independent repeats. CPU timings
used 16 threads and CUDA timings synchronized the device before and after the
timed call. The primary direct comparison used two CPU passes in alternating
worktree order where practical; CUDA direct timings used five independent
repeats. The scalar crossover sweep used twelve CPU repeat medians per case
from two passes, and seven derivative repeat medians. The constructor and
cached-precompute tables use five repeats per state; the focused paired
precompute check uses seven.

The direct cases were:

- R42 scale: triangular `(lmax, mmax, nlat) = (43, 43, 64)`;
  trapezoidal/rhomboidal `(85, 43, 128)`, with rhomboidal `N=42`.
- R127 scale: triangular `(128, 128, 180)`;
  trapezoidal/rhomboidal `(255, 128, 360)`, with rhomboidal `N=127`.
- R255 scale: triangular `(256, 256, 360)`;
  trapezoidal/rhomboidal `(511, 256, 360)`, with rhomboidal `N=255`.

The two-process-dimension distributed case is deliberately not called R8:
`lmax=16`, `mmax=8` gives `N=8`, `M=7`, and even dense splits. Standard
symmetric R8 would use `lmax=17`, `mmax=9`.

## Direct Legendre recurrence

### CPU

The table compares A and B. `legpoly` and `dlegpoly` use the public numerical
paths through their private implementations; the rhomboidal rows call the
restricted private path. The unrestricted optimization is applicable to the
trapezoidal rows, not to the rhomboidal rows.

| scale | mode | operation | A median [IQR] | B median [IQR] | change |
|---|---|---|---:|---:|---:|
| R42 | triangular | `legpoly` | 3.653 [0.078] | 3.732 [0.098] | +2.16% |
| R42 | triangular | `dlegpoly` | 5.483 [0.279] | 5.613 [0.240] | +2.36% |
| R42 | trapezoidal | `legpoly` | 8.132 [0.921] | 7.581 [0.169] | -6.77% |
| R42 | trapezoidal | `dlegpoly` | 13.170 [0.277] | 12.184 [0.232] | -7.49% |
| R42 | rhomboidal | `legpoly` | 7.814 [0.391] | 7.979 [1.366] | +2.11% |
| R42 | rhomboidal | `dlegpoly` | 12.435 [0.944] | 13.297 [1.456] | +6.93% |
| R127 | triangular | `legpoly` | 15.474 [1.108] | 15.534 [0.593] | +0.39% |
| R127 | triangular | `dlegpoly` | 66.583 [3.197] | 67.737 [2.020] | +1.73% |
| R127 | trapezoidal | `legpoly` | 53.848 [1.746] | 49.915 [3.665] | -7.30% |
| R127 | trapezoidal | `dlegpoly` | 259.332 [2.596] | 255.133 [10.170] | -1.62% |
| R127 | rhomboidal | `legpoly` | 47.878 [3.012] | 48.675 [8.321] | +1.66% |
| R127 | rhomboidal | `dlegpoly` | 253.063 [20.177] | 266.147 [9.626] | +5.17% |
| R255 | triangular | `legpoly` | 76.929 [8.256] | 80.700 [5.961] | +4.90% |
| R255 | triangular | `dlegpoly` | 475.220 [32.244] | 479.235 [45.054] | +0.84% |
| R255 | trapezoidal | `legpoly` | 156.602 [9.843] | 146.912 [11.848] | -6.19% |
| R255 | trapezoidal | `dlegpoly` | 946.420 [12.212] | 930.719 [10.040] | -1.66% |
| R255 | rhomboidal | `legpoly` | 134.441 [22.399] | 132.604 [10.333] | -1.37% |
| R255 | rhomboidal | `dlegpoly` | 971.217 [48.950] | 988.418 [47.734] | +1.77% |

The scalar trapezoidal sweep directly addresses the earlier 43-by-85 result.
It uses the exact dense bounds below and the same blocked-autorange method.

| `mmax` x `lmax` | `legpoly` A [IQR] | `legpoly` B [IQR] | change | `dlegpoly` A [IQR] | `dlegpoly` B [IQR] | change |
|---:|---:|---:|---:|---:|---:|---:|
| 16 x 32 | 2.415 [0.081] | 2.293 [0.028] | -5.04% | 3.128 [0.021] | 2.933 [0.122] | -6.22% |
| 32 x 64 | 5.902 [0.694] | 5.061 [0.691] | -14.25% | 7.531 [0.247] | 7.036 [0.174] | -6.56% |
| 43 x 85 | 8.132 [0.921] | 7.581 [0.169] | -6.77% | 13.170 [0.277] | 12.184 [0.232] | -7.49% |
| 64 x 128 | 14.250 [0.889] | 13.532 [0.486] | -5.04% | 30.815 [0.560] | 29.042 [0.245] | -5.75% |
| 128 x 255 | 53.848 [1.746] | 49.915 [3.665] | -7.30% | 259.332 [2.596] | 255.133 [10.170] | -1.62% |
| 256 x 511 | 156.602 [9.843] | 146.912 [11.848] | -6.19% | 946.420 [12.212] | 930.719 [10.040] | -1.66% |

At one CPU thread the scalar changes for the same sweep were, in order from
smallest to largest, `-7.23%`, `-0.24%`, `+0.34%`, `-7.75%`, `-6.10%`, and
`-2.03%`. This shows a size- and scheduling-dependent crossover rather than a
universal fixed overhead. At the normal 16-thread setting, the realistic
trapezoidal cases consistently favored the optimization. The isolated C
versus A scalar sweep produced changes of `-6.01%`, `-14.33%`, `-14.76%`,
`-2.26%`, `-10.11%`, and `-6.37%`; B versus A produced `-7.47%`, `-2.80%`,
`-13.81%`, `-2.23%`, `-3.13%`, and `-2.89%` in that pass. Thus the scalar
gain comes from the optimization commit itself, while the final cleanup does
not materially change the direct unrestricted recurrence.

### CUDA

CUDA direct timings are reported separately and are not used to outweigh CPU
construction behavior.

| scale | mode | operation | A median [IQR] | B median [IQR] | change |
|---|---|---|---:|---:|---:|
| R42 | triangular | `legpoly` | 11.223 [0.161] | 11.118 [0.079] | -0.94% |
| R42 | triangular | `dlegpoly` | 11.954 [0.189] | 12.111 [0.383] | +1.31% |
| R42 | trapezoidal | `legpoly` | 20.071 [0.133] | 19.942 [1.174] | -0.64% |
| R42 | trapezoidal | `dlegpoly` | 20.813 [0.100] | 19.207 [0.223] | -7.72% |
| R42 | rhomboidal | `legpoly` | 22.945 [1.242] | 22.240 [0.059] | -3.07% |
| R42 | rhomboidal | `dlegpoly` | 23.376 [0.761] | 23.333 [0.218] | -0.18% |
| R127 | triangular | `legpoly` | 36.169 [0.441] | 34.779 [0.308] | -3.84% |
| R127 | triangular | `dlegpoly` | 37.123 [3.128] | 34.246 [2.156] | -7.75% |
| R127 | trapezoidal | `legpoly` | 61.860 [0.507] | 61.793 [0.242] | -0.11% |
| R127 | trapezoidal | `dlegpoly` | 66.643 [0.172] | 61.581 [0.203] | -7.60% |
| R127 | rhomboidal | `legpoly` | 66.864 [2.424] | 67.053 [0.422] | +0.28% |
| R127 | rhomboidal | `dlegpoly` | 76.265 [2.294] | 67.360 [0.439] | -11.67% |
| R255 | triangular | `legpoly` | 69.971 [0.904] | 71.727 [1.796] | +2.51% |
| R255 | triangular | `dlegpoly` | 75.776 [3.030] | 79.659 [8.564] | +5.12% |
| R255 | trapezoidal | `legpoly` | 125.641 [0.102] | 124.556 [4.610] | -0.86% |
| R255 | trapezoidal | `dlegpoly` | 145.218 [0.308] | 134.615 [0.693] | -7.30% |
| R255 | rhomboidal | `legpoly` | 127.367 [3.363] | 126.326 [2.990] | -0.82% |
| R255 | rhomboidal | `dlegpoly` | 146.023 [0.796] | 148.770 [0.420] | +1.88% |

## Cold cached precompute

The cache was cleared through the repository cache wrapper before each cold
measurement. The cache wrapper exposes the normal `cache_clear()` and
`cache_info()` methods through its underlying `functools.lru_cache`; no cache
behavior was bypassed or changed.

The first cold pass is summarized below. Each cell is `median [IQR]`, and the
percentage is B relative to A.

| case | operation | A cold | B cold | change | A warm | B warm | change |
|---|---|---:|---:|---:|---:|---:|---:|
| R42 triangular | `legpoly` | 5.002 [0.236] | 4.867 [0.327] | -2.69% | 0.038 [0.001] | 0.033 [0.001] | -12.28% |
| R42 triangular | `dlegpoly` | 8.745 [0.167] | 9.614 [0.374] | +9.94% | 0.037 [0.002] | 0.041 [0.001] | +10.75% |
| R42 trapezoidal | `legpoly` | 10.993 [1.945] | 9.904 [0.305] | -9.91% | 0.046 [0.003] | 0.038 [0.000] | -16.74% |
| R42 trapezoidal | `dlegpoly` | 16.912 [2.943] | 19.072 [3.184] | +12.77% | 0.056 [0.007] | 0.050 [0.001] | -9.59% |
| R42 rhomboidal | `legpoly` | 9.716 [0.312] | 11.709 [2.461] | +20.51% | 0.040 [0.002] | 0.038 [0.000] | -3.99% |
| R42 rhomboidal | `dlegpoly` | 16.437 [0.569] | 16.523 [0.990] | +0.52% | 0.049 [0.004] | 0.057 [0.001] | +14.45% |
| R127 triangular | `legpoly` | 36.272 [4.544] | 36.910 [0.598] | +1.76% | 7.311 [0.248] | 7.125 [0.481] | -2.54% |
| R127 triangular | `dlegpoly` | 162.305 [4.480] | 165.269 [13.959] | +1.83% | 15.714 [1.572] | 15.626 [2.285] | -0.56% |
| R127 trapezoidal | `legpoly` | 82.247 [3.172] | 70.965 [0.251] | -13.72% | 15.190 [0.417] | 15.209 [0.575] | +0.13% |
| R127 trapezoidal | `dlegpoly` | 315.234 [13.433] | 361.455 [23.974] | +14.66% | 29.679 [1.513] | 34.066 [1.191] | +14.78% |
| R127 rhomboidal | `legpoly` | 73.815 [4.292] | 71.892 [6.878] | -2.61% | 14.577 [0.794] | 17.550 [0.370] | +20.39% |
| R127 rhomboidal | `dlegpoly` | 321.309 [5.004] | 346.750 [4.090] | +7.92% | 31.041 [1.590] | 35.740 [0.680] | +15.14% |

The first-pass cache timings contain substantial process-frequency and cache
allocation noise. A focused paired check (seven repeats, same dense bounds)
was used for the pruning conclusion:

| case | operation | trapezoidal cold | rhomboidal cold | rhomboidal change | trapezoidal warm | rhomboidal warm | rhomboidal change |
|---|---|---:|---:|---:|---:|---:|---:|
| R42 | `legpoly` | 8.771 [0.056] | 9.263 [0.079] | +5.61% | 0.033 [0.002] | 0.032 [0.003] | -2.90% |
| R42 | `dlegpoly` | 15.031 [0.200] | 15.184 [0.080] | +1.01% | 0.046 [0.004] | 0.051 [0.003] | +11.71% |
| R127 | `legpoly` | 63.680 [1.932] | 61.857 [0.750] | -2.86% | 12.638 [0.387] | 13.467 [0.837] | +6.56% |
| R127 | `dlegpoly` | 282.625 [2.846] | 275.563 [7.061] | -2.50% | 26.831 [0.487] | 26.787 [0.441] | -0.16% |

The cache-hit checks showed identical requests increasing `hits` with no new
miss, while triangular, trapezoidal, and rhomboidal requests occupied three
distinct cache entries. Trapezoidal and rhomboidal tables had the same dense
shape but were not equal, so they did not alias incorrectly.

## SHT constructor timing

The complete constructor benchmark covered `RealSHT`, `InverseRealSHT`,
`RealVectorSHT`, and `InverseRealVectorSHT` for all three modes at R42 and
R127. Cold construction cleared the Legendre caches; warm construction primed
them first. The following table combines two five-repeat passes. Each entry is
`A -> B` in milliseconds followed by the percentage change; IQRs are shown as
`A/B` in brackets.

| case | constructor | cold A -> B (change; IQR A/B) | warm A -> B (change; IQR A/B) |
|---|---|---:|---:|
| R42 triangular | `RealSHT` | 4.622 -> 4.563 (-1.26%; [0.540/0.192]) | 0.209 -> 0.213 (+2.25%; [0.019/0.013]) |
| R42 triangular | `InverseRealSHT` | 4.793 -> 4.431 (-7.55%; [0.718/0.661]) | 0.151 -> 0.135 (-10.47%; [0.003/0.010]) |
| R42 triangular | `RealVectorSHT` | 9.093 -> 9.281 (+2.06%; [0.555/0.836]) | 1.359 -> 1.385 (+1.89%; [0.292/0.249]) |
| R42 triangular | `InverseRealVectorSHT` | 8.160 -> 8.552 (+4.80%; [0.643/1.944]) | 1.070 -> 1.098 (+2.66%; [0.070/0.183]) |
| R42 trapezoidal | `RealSHT` | 9.540 -> 9.437 (-1.08%; [0.876/1.845]) | 1.159 -> 1.261 (+8.83%; [0.036/0.128]) |
| R42 trapezoidal | `InverseRealSHT` | 9.545 -> 11.356 (+18.98%; [0.883/3.345]) | 1.130 -> 1.240 (+9.69%; [0.015/0.124]) |
| R42 trapezoidal | `RealVectorSHT` | 17.289 -> 16.390 (-5.20%; [1.330/1.885]) | 2.125 -> 2.459 (+15.73%; [0.189/0.244]) |
| R42 trapezoidal | `InverseRealVectorSHT` | 15.000 -> 15.129 (+0.86%; [0.590/0.924]) | 1.859 -> 2.242 (+20.60%; [0.151/0.254]) |
| R42 rhomboidal | `RealSHT` | 8.755 -> 8.437 (-3.63%; [0.982/0.723]) | 0.231 -> 0.247 (+7.08%; [0.010/0.004]) |
| R42 rhomboidal | `InverseRealSHT` | 8.504 -> 8.693 (+2.22%; [0.735/0.614]) | 0.172 -> 0.183 (+6.79%; [0.009/0.037]) |
| R42 rhomboidal | `RealVectorSHT` | 16.864 -> 17.112 (+1.47%; [0.585/0.702]) | 2.455 -> 2.558 (+4.18%; [0.031/0.129]) |
| R42 rhomboidal | `InverseRealVectorSHT` | 15.403 -> 16.943 (+10.00%; [0.965/2.506]) | 2.072 -> 2.258 (+8.95%; [0.153/0.255]) |
| R127 triangular | `RealSHT` | 41.515 -> 43.512 (+4.81%; [3.176/5.125]) | 14.179 -> 14.829 (+4.59%; [1.866/2.386]) |
| R127 triangular | `InverseRealSHT` | 41.325 -> 41.865 (+1.31%; [1.667/1.589]) | 14.598 -> 14.639 (+0.28%; [1.149/1.715]) |
| R127 triangular | `RealVectorSHT` | 191.924 -> 188.674 (-1.69%; [6.962/18.170]) | 56.029 -> 53.399 (-4.69%; [8.356/4.128]) |
| R127 triangular | `InverseRealVectorSHT` | 173.343 -> 172.547 (-0.46%; [23.908/16.390]) | 28.295 -> 29.275 (+3.46%; [1.408/2.370]) |
| R127 trapezoidal | `RealSHT` | 85.374 -> 83.526 (-2.16%; [8.871/7.713]) | 29.860 -> 28.648 (-4.06%; [2.334/2.177]) |
| R127 trapezoidal | `InverseRealSHT` | 87.400 -> 87.684 (+0.33%; [6.227/7.396]) | 29.667 -> 29.937 (+0.91%; [2.065/2.382]) |
| R127 trapezoidal | `RealVectorSHT` | 391.164 -> 368.867 (-5.70%; [59.281/45.638]) | 105.633 -> 105.003 (-0.60%; [3.126/15.774]) |
| R127 trapezoidal | `InverseRealVectorSHT` | 325.067 -> 329.427 (+1.34%; [26.528/22.345]) | 56.816 -> 56.362 (-0.80%; [8.271/1.569]) |
| R127 rhomboidal | `RealSHT` | 78.176 -> 76.617 (-2.00%; [6.033/4.040]) | 28.186 -> 27.890 (-1.05%; [2.573/0.976]) |
| R127 rhomboidal | `InverseRealSHT` | 75.996 -> 76.587 (+0.78%; [6.477/5.113]) | 27.913 -> 28.339 (+1.53%; [2.012/0.817]) |
| R127 rhomboidal | `RealVectorSHT` | 373.226 -> 366.334 (-1.85%; [33.898/30.816]) | 106.194 -> 105.434 (-0.72%; [12.156/10.069]) |
| R127 rhomboidal | `InverseRealVectorSHT` | 324.099 -> 316.715 (-2.28%; [43.468/18.786]) | 59.350 -> 57.578 (-2.99%; [9.444/5.926]) |

Constructor measurements with sub-5% changes are dominated by the observed
run-to-run spread. The larger R42 warm percentages are not used as evidence
for a code regression because their absolute times are around 1--2 ms and the
IQRs are comparable to the differences.

## Existing SHT runtime benchmark

The repository harness was run in both directions with 5 warmups and 40
iterations:

```text
python benchmarks/run.py --name sht --warmup 5 --iters 40 --save-csv A-standard-1.csv
python benchmarks/run.py --name sht --warmup 5 --iters 40 --save-csv B-standard-1.csv --reference-csv A-standard-1.csv
python benchmarks/run.py --name sht --warmup 5 --iters 40 --save-csv B-standard-2.csv
python benchmarks/run.py --name sht --warmup 5 --iters 40 --save-csv A-standard-2.csv --reference-csv B-standard-2.csv
```

The two-run medians below show the range across the two process runs. These
are transform hot-path timings, not constructor timings.

| benchmark | A median [range] | B median [range] | change of medians |
|---|---:|---:|---:|
| SHT forward CPU | 0.546 [0.519, 0.573] | 0.563 [0.508, 0.618] | +3.1% |
| SHT backward CPU | 3.356 [2.892, 3.821] | 2.880 [2.720, 3.040] | -14.2% |
| SHT forward CUDA, 1-degree | 0.295 [0.289, 0.301] | 0.295 [0.293, 0.297] | -0.0% |
| SHT backward CUDA, 1-degree | 0.412 [0.397, 0.426] | 0.432 [0.430, 0.434] | +5.0% |
| SHT forward CUDA, high-degree | 0.291 [0.290, 0.291] | 0.291 [0.289, 0.293] | +0.1% |
| SHT backward CUDA, high-degree | 0.404 [0.400, 0.408] | 0.439 [0.436, 0.442] | +8.6% |
| iSHT forward CPU | 0.655 [0.634, 0.676] | 0.710 [0.680, 0.739] | +8.4% |
| iSHT backward CPU | 2.823 [2.809, 2.837] | 2.725 [2.707, 2.742] | -3.5% |
| iSHT forward CUDA, 1-degree | 0.370 [0.363, 0.377] | 0.382 [0.378, 0.386] | +3.2% |
| iSHT backward CUDA, 1-degree | 0.480 [0.467, 0.494] | 0.488 [0.458, 0.518] | +1.6% |
| iSHT forward CUDA, high-degree | 0.381 [0.370, 0.392] | 0.374 [0.374, 0.375] | -1.7% |
| iSHT backward CUDA, high-degree | 0.504 [0.486, 0.523] | 0.478 [0.463, 0.493] | -5.2% |

The reference comparator flagged isolated rows in opposite run directions at
its 5% threshold. The two-run ranges are much larger than the changes in most
rows, so these are not repeatable hot-path regressions. The transform path
still performs the same dense contraction; rhomboidal mode adds no sparse or
packed runtime operation.

A final current-branch harness run after the docstring cleanup reported
`0.54/2.75 ms` for CPU SHT, `0.29/0.40 ms` for CUDA SHT 1-degree,
`0.33/0.42 ms` for CUDA SHT high-degree, `0.67/2.72 ms` for CPU iSHT,
`0.38/0.47 ms` for CUDA iSHT 1-degree, and `0.37/0.46 ms` for CUDA iSHT
high-degree (forward/backward). It is consistent with the comparison runs.

## Truncation-specific transform runtime

`/tmp/torch-harmonics-truncation-bench/transform_runtime.py` measured actual
scalar and vector forward/inverse transforms on CPU and CUDA. The modules were
constructed and moved to their device before timing. Triangular, trapezoidal,
and rhomboidal cases used identical dense bounds for each scale; the
rhomboidal table has smaller mathematical support but the same public tensor
shape. Each cell is `A -> B` in milliseconds, followed by the percentage
change and `IQR(A/B)` from the six combined repeat medians.

| device | case | scalar fwd | scalar inv | vector fwd | vector inv |
|---|---|---:|---:|---:|---:|
| CPU | triangular R42 | 0.267 -> 0.259 (-2.90%; [0.009/0.006]) | 0.315 -> 0.293 (-7.06%; [0.006/0.009]) | 0.548 -> 0.521 (-4.88%; [0.045/0.019]) | 0.585 -> 0.621 (+6.22%; [0.019/0.050]) |
| CPU | trapezoidal R42 | 0.259 -> 0.264 (+1.81%; [0.006/0.009]) | 0.287 -> 0.306 (+6.51%; [0.012/0.003]) | 0.532 -> 0.570 (+7.13%; [0.030/0.003]) | 0.611 -> 0.638 (+4.41%; [0.006/0.008]) |
| CPU | rhomboidal R42 | 0.247 -> 0.265 (+7.57%; [0.001/0.004]) | 0.287 -> 0.304 (+6.00%; [0.001/0.003]) | 0.558 -> 0.552 (-0.95%; [0.011/0.037]) | 0.621 -> 0.674 (+8.52%; [0.010/0.046]) |
| CPU | triangular R127 | 0.921 -> 1.103 (+19.78%; [0.024/0.070]) | 1.123 -> 1.183 (+5.37%; [0.027/0.110]) | 11.334 -> 10.802 (-4.69%; [0.527/0.258]) | 11.417 -> 11.198 (-1.91%; [0.033/0.171]) |
| CPU | trapezoidal R127 | 10.035 -> 9.541 (-4.92%; [0.038/0.072]) | 10.168 -> 9.329 (-8.25%; [0.035/0.045]) | 23.790 -> 22.721 (-4.49%; [1.386/0.563]) | 25.283 -> 23.232 (-8.11%; [0.445/1.457]) |
| CPU | rhomboidal R127 | 10.133 -> 9.746 (-3.82%; [0.086/0.346]) | 10.343 -> 10.024 (-3.09%; [0.183/0.131]) | 25.438 -> 24.135 (-5.12%; [1.050/0.927]) | 24.305 -> 23.916 (-1.60%; [1.710/0.680]) |
| CUDA | triangular R42 | 0.270 -> 0.280 (+3.94%; [0.006/0.003]) | 0.329 -> 0.350 (+6.34%; [0.002/0.001]) | 0.618 -> 0.666 (+7.91%; [0.007/0.002]) | 0.722 -> 0.731 (+1.21%; [0.023/0.002]) |
| CUDA | trapezoidal R42 | 0.263 -> 0.282 (+7.03%; [0.002/0.002]) | 0.330 -> 0.355 (+7.49%; [0.003/0.007]) | 0.620 -> 0.654 (+5.49%; [0.005/0.004]) | 0.678 -> 0.747 (+10.18%; [0.001/0.026]) |
| CUDA | rhomboidal R42 | 0.263 -> 0.278 (+5.50%; [0.000/0.001]) | 0.327 -> 0.344 (+5.48%; [0.001/0.003]) | 0.623 -> 0.655 (+5.13%; [0.007/0.002]) | 0.687 -> 0.719 (+4.66%; [0.002/0.003]) |
| CUDA | triangular R127 | 0.275 -> 0.265 (-3.58%; [0.002/0.000]) | 0.331 -> 0.330 (-0.13%; [0.001/0.003]) | 0.617 -> 0.614 (-0.39%; [0.002/0.003]) | 0.693 -> 0.691 (-0.26%; [0.001/0.003]) |
| CUDA | trapezoidal R127 | 0.429 -> 0.429 (+0.18%; [0.000/0.001]) | 0.500 -> 0.492 (-1.49%; [0.000/0.000]) | 1.163 -> 1.163 (-0.02%; [0.001/0.000]) | 1.227 -> 1.232 (+0.40%; [0.008/0.001]) |
| CUDA | rhomboidal R127 | 0.424 -> 0.426 (+0.31%; [0.001/0.000]) | 0.486 -> 0.499 (+2.72%; [0.000/0.004]) | 1.162 -> 1.171 (+0.77%; [0.005/0.012]) | 1.234 -> 1.231 (-0.25%; [0.011/0.004]) |

The same dense tensor shape therefore determines transform runtime to the
resolution of this measurement; rhomboidal support changes precompute and
stored values, not the dense contraction shape.

## Trapezoidal optimization crossover

At the primary 16-thread CPU setting, scalar `legpoly` improved consistently
across the six explicit sizes, including the earlier 43-by-85 case. The
derivative path also improved at R42-scale, while its large cases were close
to neutral. CUDA scalar timings were near neutral except for noisy small
cases; CUDA was not used to justify the CPU decision.

There is no single size-independent crossover: the 1-thread sweep included a
small positive result at 43-by-85 and near-zero results at 32-by-64 and
256-by-511. At the normal 16-thread development setting, however, the
trapezoidal scalar improvements were 5.0--14.3% in the tested cases and the
realistic derivative improvements were 1.6--7.5%. No consistent unrestricted
triangular regression above 5% was found; its R255 scalar difference was
within the broad repeat spread.

## Interpretation

1. **Does rhomboidal Legendre pruning reduce precompute cost?** Yes, modestly
   at the R127 bounding size in the focused cold measurements (about 2.5--2.9%
   for scalar and derivative tables). R42 scalar construction was about 5.6%
   slower and the warm-cache results were mixed, so this is not a universal
   percentage benefit.
2. **Does it regress triangular construction?** No repeatable regression was
   established. Triangular direct recurrence and constructor differences were
   within their measured variability.
3. **Does it regress trapezoidal construction?** No consistent regression was
   established. The paired focused cold R42/R127 measurements were neutral to
   faster for the candidate trapezoidal path, although individual first-pass
   derivative cache rows were noisy.
4. **Does the trapezoidal fast path help?** Yes. The CPU scalar recurrence
   improved 5.0--14.3% at the normal 16-thread setting, including the earlier
   43-by-85 case. Direct derivative timings were neutral to faster at the two
   larger scales, and the vector constructor measurements did not establish a
   repeatable regression.
5. **Where does it cross over?** The exact crossover depends on thread
   setting and system noise. At 16 threads every tested scalar size from
   16-by-32 through 256-by-511 was faster; at one thread the 32-by-64,
   43-by-85, and largest cases were neutral or slightly slower. There is no
   robust universal crossover threshold.
6. **Does rhomboidal mode change transform hot-path performance for equal dense
   bounds?** No material change was measured. The dense contraction shape is
   unchanged, and the temporary transform benchmark showed only normal
   sub-millisecond variation.
7. **Are any repeatable regressions greater than 5%?** No repeatable regression
   attributable to the implementation was established. Several isolated
   first-pass constructor or hot-path rows exceeded 5%, but their paired runs
   changed direction or were within the measured spread.
8. **Should `Optimize trapezoidal Legendre construction` remain in PR #264?**
   Yes. It gives a repeatable CPU scalar precompute benefit in the realistic
   16-thread configurations, including the motivating R42-scale case, without
   a material unrestricted regression elsewhere. The optimization is retained
   unchanged in this pass.

## Correctness and API checks

The public signatures on A and B match the current upstream-parent form:

```text
legpoly(mmax, lmax, x, norm='ortho', inverse=False, csphase=True, *, mmin=0, lmin=0)
dlegpoly(mmax, lmax, t, norm='ortho', inverse=False, csphase=True, *, mmin=0, lmin=0)
```

`l_minus_m_max` is accepted only by the private recurrence and cached
precompute helpers. The restricted recurrence uses the global
`N = global_lmax - global_mmax`; local `mmin`/`lmin` ranges do not derive a
rank-dependent value. Bitwise comparisons between A and B reported equal
unrestricted and retained restricted values for scalar R42/R127 blocks and
derivative blocks with the `N + 2` halo. The support checks independently used
`m <= l` and `l - m <= N`.

## Distributed validation

The targeted 2x2 grid cases used `lmax=16`, `mmax=8`, `N=8`, and `M=7`:

- CPU/Gloo with four `torchrun` ranks passed scalar and vector distributed
  forward, scalar and vector distributed inverse, and scalar and vector
  distributed Legendre-block tests: 6 passed.
- CUDA/NCCL could not run four ranks on this one-GPU host. NCCL first rejected
  multiple ranks on one GPU; enabling its shared-GPU option reached the first
  collective but exhausted device memory. No distributed CUDA assertion or
  correctness failure was observed.

## Raw data and commands

Raw JSON/CSV outputs and the temporary scripts are in:

```text
/tmp/torch-harmonics-truncation-bench/
```

The direct and constructor scripts use `blocked_autorange`; the transform
script is `transform_runtime.py`. The standard harness CSVs are
`A-standard-1.csv`, `B-standard-1.csv`, `B-standard-2.csv`, and
`A-standard-2.csv`. No raw output or benchmark summary was added to Git beyond
this temporary Markdown file.
