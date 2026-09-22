# Serial layers

## Spherical harmonic transforms

All four transforms accept independent non-inclusive `lmax` and `mmax` limits
and an optional `lmmax` bandwidth. The distributed transforms use the same
limits. Retained modes satisfy `m < mmax` and `m <= l < lmax`; when `lmmax` is
provided, they also satisfy `l - m < lmmax`. With `lmmax=None`, there is no
additional degree-minus-order restriction.

These limits support generalized pentagonal truncation. Equal `lmax` and
`mmax` with no bandwidth cap give triangular bounds, while independent limits
with `lmmax=None` give trapezoidal bounds. A standard rhomboid has
`lmmax == mmax` and `lmax == mmax + lmmax - 1`. For example:

```python
from torch_harmonics import RealSHT

# Standard R42: l - m == 42 is retained; l - m == 43 is excluded.
sht = RealSHT(128, 256, lmax=85, mmax=43, lmmax=43, grid="legendre-gauss")

# The same order limit and bandwidth, capped independently at l < 64.
capped = RealSHT(128, 256, lmax=64, mmax=43, lmmax=43, grid="legendre-gauss")
```

Omitted limits use the latitude grid's default degree limit and the longitude
Nyquist limit, with `mmax` capped at `lmax`. Explicit `mmax > lmax` raises
`ValueError` when `lmmax` is provided. `truncate_sht` returns an immutable
truncation descriptor carrying `lmax`, `mmax`, and `lmmax`, used by each
transform.

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   RealSHT
   InverseRealSHT
   RealVectorSHT
   InverseRealVectorSHT
```

## Convolutions

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   SpectralConvS2
   DiscreteContinuousConvS2
   DiscreteContinuousConvTransposeS2
```

## Filter basis

```{eval-rst}
.. currentmodule:: torch_harmonics.filter_basis

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_filter_basis
   FilterBasis
   PiecewiseLinearFilterBasis
   HarmonicFilterBasis
   ZernikeFilterBasis
   FourierBesselFilterBasis
```

## Attention mechanism

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   AttentionS2
   NeighborhoodAttentionS2
```

## Resampling and quadrature

```{eval-rst}
.. currentmodule:: torch_harmonics

.. autosummary::
   :toctree: generated
   :nosignatures:

   ResampleS2
   QuadratureS2
```

## Random fields

```{eval-rst}
.. currentmodule:: torch_harmonics.random_fields

.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussianRandomFieldS2
```
