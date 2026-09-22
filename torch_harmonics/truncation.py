# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import warnings
from numbers import Integral
from typing import NamedTuple, Optional


class SHTTruncation(NamedTuple):
    """Global non-inclusive degree, order, and optional degree-minus-order limits."""

    lmax: int
    mmax: int
    lmmax: Optional[int] = None


def _truncate_lmax(nlat: int, grid: Optional[str] = "equiangular") -> int:
    """
    Truncate the maximum spherical harmonic degree based on the latitude grid. The maximum degree
    corresponds to the maximum degree of associated Legendre polynomials that can be square-integrated
    exactly.

    | Grid Type           | Includes Poles? | Exactness       | Heuristic ($L_{\text{max}}$) |
    | :---                | :---:           | :---:           | :---:                        |
    | Legendre-Gauss (GL) | No              | $2N - 1$        | $N - 1$                      |
    | Gauss-Lobatto (GLL) | Yes             | $2N - 3$        | $N - 2$                      |
    | Equiangular (CC)    | Yes             | $\approx N - 1$ | $\approx N/2$                |

    Parameters
    ----------
    nlat : int
        Number of latitude points
    grid : str, optional
        Grid type (``"legendre-gauss"``, ``"lobatto"``, ``"equiangular"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``

    Returns
    -------
    int
        Maximum spherical harmonic degree (non-inclusive)
    """
    if grid == "legendre-gauss":
        return nlat
    elif grid == "lobatto":
        return nlat - 1
    elif grid in ["equiangular", "equiangular-trapezoidal"]:
        warnings.warn(
            "Default SHT truncation changed in v0.9.0: equiangular/equiangular-trapezoidal grids now truncate to (nlat+1)//2. " "Specify lmax explicitly to override.",
            UserWarning,
            stacklevel=2,
        )
        return (nlat + 1) // 2
    else:
        raise ValueError(f"Unknown grid type {grid}")


def _truncate_mmax(nlon: int) -> int:
    """
    Truncate the maximum azimuthal harmonic degree based on the longitude grid. This is the same as the
    Nyquist frequency.

    Parameters
    ----------
    nlon : int
        Number of longitude points

    Returns
    -------
    int
        Maximum azimuthal harmonic degree (non-inclusive)
    """
    return nlon // 2 + 1


def truncate_sht(nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None, grid: Optional[str] = "equiangular", lmmax: Optional[int] = None) -> SHTTruncation:
    r"""
    Determine the maximum spherical harmonic degree and order for an SHT based
    on the spatial grid.

    When ``lmax`` or ``mmax`` are not provided, they are inferred from the grid
    resolution.  The default truncation for each grid type is chosen so that the
    associated Legendre polynomials up to the returned degree can be
    square-integrated exactly by the corresponding quadrature rule:

    .. list-table:: Default latitudinal truncation :math:`l_{\max}` for :math:`N_\theta` latitude points
       :header-rows: 1
       :widths: 30 15 25 30

       * - Grid type
         - Includes poles?
         - Quadrature exactness
         - Default :math:`l_{\max}`
       * - ``"legendre-gauss"``
         - No
         - :math:`2 N_\theta - 1`
         - :math:`N_\theta`
       * - ``"lobatto"``
         - Yes
         - :math:`2 N_\theta - 3`
         - :math:`N_\theta - 1`
       * - ``"equiangular"`` / ``"equiangular-trapezoidal"``
         - Yes
         - :math:`\approx N_\theta - 1`
         - :math:`\lfloor (N_\theta + 1) / 2 \rfloor`

    The default longitudinal truncation is the Nyquist limit of the uniform
    longitude grid: :math:`m_{\max} = \lfloor N_\lambda / 2 \rfloor + 1`.

    Degree and order limits are independent. Orders are capped at ``lmax`` because
    physical modes satisfy ``m <= l``. An explicit ``mmax > lmax`` is an error
    when ``lmmax`` is provided; bandwidth-constrained limits are not reshaped.
    Omitted ``mmax`` defaults to the smaller of the Nyquist limit and ``lmax``.

    Modes satisfy ``m < mmax`` and ``m <= l < lmax``, and additionally
    ``l - m < lmmax`` when the bandwidth is provided. These limits describe
    generalized pentagonal truncation: equal ``lmax`` and ``mmax`` without a
    bandwidth cap give triangular bounds; ``lmmax=None`` gives trapezoidal
    bounds; ``lmmax == mmax`` with ``lmax == mmax + lmmax - 1`` gives a standard
    rhomboid. A smaller independent ``lmax`` caps that rhomboid.

    Parameters
    ----------
    nlat : int
        Number of latitude points :math:`N_\theta`.
    nlon : int
        Number of longitude points :math:`N_\lambda`.
    lmax : int, optional
        User-defined maximum spherical harmonic degree (non-inclusive).
        If not provided, the maximum degree is determined from the latitude
        grid as shown in the table above.
    mmax : int, optional
        User-defined maximum azimuthal harmonic order (non-inclusive).
        If not provided, set to the smaller of ``lmax`` and the Nyquist limit
        :math:`\lfloor N_\lambda / 2 \rfloor + 1`.
    grid : str, optional
        Grid type (``"legendre-gauss"``, ``"lobatto"``, ``"equiangular"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``.
    lmmax : int, optional
        Maximum non-inclusive degree-minus-order bandwidth. Modes satisfy
        ``l - m < lmmax`` when provided. If ``None``, no additional
        degree-minus-order restriction is applied.

    Returns
    -------
    SHTTruncation
        Immutable ``(lmax, mmax, lmmax)`` limits, all non-inclusive.

    Examples
    --------
    >>> from torch_harmonics import truncate_sht
    >>> truncate_sht(128, 256, grid="legendre-gauss")
    SHTTruncation(lmax=128, mmax=128, lmmax=None)
    >>> truncate_sht(128, 256, grid="lobatto")
    SHTTruncation(lmax=127, mmax=127, lmmax=None)
    >>> truncate_sht(128, 256, grid="equiangular")
    SHTTruncation(lmax=64, mmax=64, lmmax=None)
    >>> truncate_sht(128, 256, lmax=96, mmax=48, grid="legendre-gauss")
    SHTTruncation(lmax=96, mmax=48, lmmax=None)
    >>> # Standard R42 retains l - m == 42 and excludes l - m == 43.
    >>> truncate_sht(128, 256, lmax=85, mmax=43, grid="legendre-gauss", lmmax=43)
    SHTTruncation(lmax=85, mmax=43, lmmax=43)
    >>> # The same bandwidth and order limit, with an independent degree cap.
    >>> truncate_sht(128, 256, lmax=64, mmax=43, grid="legendre-gauss", lmmax=43)
    SHTTruncation(lmax=64, mmax=43, lmmax=43)
    """

    explicit_mmax = mmax is not None
    lmax = _truncate_lmax(nlat, grid) if lmax is None else lmax
    mmax = _truncate_mmax(nlon) if mmax is None else mmax

    for name, limit in (("lmax", lmax), ("mmax", mmax), ("lmmax", lmmax)):
        if name == "lmmax" and limit is None:
            continue
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0:
            raise ValueError(f"{name} must be a positive integer, got {limit!r}")

    if lmmax is not None and explicit_mmax and mmax > lmax:
        raise ValueError("mmax must not exceed lmax when lmmax is provided")
    mmax = min(mmax, lmax)

    return SHTTruncation(lmax, mmax, lmmax)
