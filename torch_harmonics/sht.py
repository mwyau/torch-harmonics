# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

import torch
import torch.nn as nn

from torch_harmonics.fft import irfft, rfft
from torch_harmonics.legendre import _precompute_dlegpoly, _precompute_legpoly
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights, lobatto_weights
from torch_harmonics.truncation import truncate_sht
from torch_harmonics.utils import check


def _periodic_latitude_extension(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    r"""Extend ``(..., m, nlat)`` data to a periodic meridian.

    The caller supplies the continuation signs, for example
    :math:`(-1)^m` for scalar modes and :math:`(-1)^{m+1}` for tangential
    vector components. The two poles occur once in the returned sequence; only
    the interior latitude rings are reversed. ``signs`` has shape ``(m, 1)``
    and is broadcast over leading dimensions. It may be a floating-point or
    integer tensor; multiplication preserves the complex dtype of ``x``.
    """

    interior_reversed = x[..., 1:-1].flip(-1)
    return torch.cat((x, signs * interior_reversed), dim=-1)


def _periodic_latitude_extension_adjoint(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    r"""Apply the Hilbert adjoint of :func:`_periodic_latitude_extension`.

    The mirrored interior rings in the periodic extension contribute to the
    corresponding original rings under the adjoint. The two poles are not
    doubled because each occurs only once in the extension.
    """

    nlat = x.shape[-1] // 2 + 1
    interior = x[..., 1 : nlat - 1] + signs * x[..., nlat:].flip(-1)
    return torch.cat((x[..., :1], interior, x[..., nlat - 1 : nlat]), dim=-1)


def _fourier_shift_latitude(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Shift a periodic meridian by half a sample in Fourier space.

    ``phase`` contains ``exp(+i*pi*k/M)`` for the signed FFT frequency
    representatives ``k`` of a length-``M`` transform. In particular, the
    even-length Nyquist mode is represented once at ``k=-M/2``, matching the
    centered zero-padding convention used by the original dense path.
    """

    spectrum = torch.fft.fft(x, dim=-1, norm="forward")
    return torch.fft.ifft(spectrum * phase, dim=-1, norm="forward")


def _fourier_shift_latitude_adjoint(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Apply the complex Hilbert adjoint of :func:`_fourier_shift_latitude`."""

    spectrum = torch.fft.fft(x, dim=-1, norm="forward")
    return torch.fft.ifft(spectrum * phase.conj(), dim=-1, norm="forward")


def _fold_resampled_latitude(
    x: torch.Tensor,
    signs: torch.Tensor,
    quadrature_weights: torch.Tensor,
    midpoint_weights: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    r"""Fold weighted midpoint rings back onto the original latitude grid.

    For resampled equiangular analysis on the pole-including Clenshaw--Curtis
    grid, this applies
    the Appendix-A folded-ring operator

    ``Q_e x + A* Q_o A x``

    where ``A`` extends the original meridian, shifts it by half a sample, and
    selects the first ``N-1`` midpoint rings. ``A*`` is implemented with the
    conjugate Fourier phase, zero-padding in the selected-ring adjoint, and the
    explicit adjoint of the periodic extension. This is the algebraic form of
    the folded-ring optimization described in Reinecke, Belkner & Carron
    (2023), Appendix A, DOI: 10.1051/0004-6361/202346717.
    """

    extended = _periodic_latitude_extension(x, signs)
    midpoint = _fourier_shift_latitude(extended, phase)[..., : x.shape[-1] - 1]
    midpoint = midpoint * midpoint_weights
    midpoint = torch.cat((midpoint, torch.zeros_like(midpoint)), dim=-1)
    folded = _fourier_shift_latitude_adjoint(midpoint, phase)
    folded = _periodic_latitude_extension_adjoint(folded, signs)
    return quadrature_weights * x + folded


def _precompute_resampled_projection(
    weights: torch.Tensor,
    signs: torch.Tensor,
    quadrature_weights: torch.Tensor,
    midpoint_weights: torch.Tensor,
    phase: torch.Tensor,
    row_chunk: int = 4,
) -> torch.Tensor:
    """Fold projection rows into an effective original-ring projection."""

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")

    rows = weights.transpose(-3, -2)
    effective_rows = torch.empty(rows.shape, dtype=phase.dtype, device=rows.device)

    for start in range(0, rows.shape[-3], row_chunk):
        size = min(row_chunk, rows.shape[-3] - start)
        rows_chunk = rows.narrow(-3, start, size)
        folded = _fold_resampled_latitude(
            rows_chunk,
            signs,
            quadrature_weights,
            midpoint_weights,
            phase,
        )
        # U is Hermitian, so P U = conj(U @ P.T).T.
        effective_rows.narrow(-3, start, size).copy_(folded.conj())

    return effective_rows.transpose(-3, -2).contiguous()


def _resolve_sht_limits(nlat, nlon, lmax, mmax, grid):
    """Resolve upstream triangular limits and validate resampled analysis."""

    direct_lmax = (nlat + 1) // 2 if grid == "equiangular" else None

    # An explicit degree beyond the direct quadrature limit with no order limit
    # should use the recoverable longitude order before the upstream triangular
    # truncation is applied.
    # All other calls retain the exact upstream arguments and defaults.
    if grid == "equiangular" and lmax is not None and lmax > direct_lmax and mmax is None:
        mmax = (nlon - 1) // 2 + 1

    lmax, mmax = truncate_sht(nlat, nlon, lmax, mmax, grid)

    if grid == "equiangular" and lmax > direct_lmax:
        max_lmax = nlat - 1
        max_mmax = (nlon - 1) // 2 + 1
        if lmax > max_lmax or mmax > max_mmax:
            raise ValueError(
                f"Equiangular SHT analysis on a {nlat}x{nlon} grid supports exclusive " f"lmax <= {max_lmax}, mmax <= {max_mmax}; resolved limits were ({lmax}, {mmax})"
            )

    return lmax, mmax


class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input.

    Given a real-valued signal :math:`f(\theta, \lambda)` sampled on the sphere,
    the forward scalar SHT computes the spherical harmonic coefficients via a
    longitudinal FFT followed by Legendre quadrature:

    .. math::

        \hat{f}_l^m = 2\pi \sum_{k=0}^{N_\theta - 1}
            \tilde{f}_m(\theta_k)\, P_l^m(\cos\theta_k)\, q_k

    where :math:`\tilde{f}_m` are the Fourier modes and :math:`q_k` are the
    quadrature weights.

    On the pole-including equiangular grid, omitted limits retain the
    conservative direct-quadrature default.  When an explicit ``lmax`` exceeds
    that limit, the forward transform uses meridional resampling and folding
    for analysis beyond the direct quadrature limit, allowing recovery up to
    the grid's supported limits.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    precompute_resampling : bool
        For resampled equiangular analysis, precompute the latitude-resampling
        operator into the projection weights. This increases persistent projection
        storage and construction cost but reduces per-forward work and temporary
        memory. Has no effect when latitude resampling is not required. By default
        ``False``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> sht = th.RealSHT(nlat, nlon)
    >>> signal = torch.randn(1, nlat, nlon)
    >>> coeffs = sht(signal)   # shape (1, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 64, 64])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.rfft`) to compute the
        longitudinal Fourier transform efficiently.  When running in **float16** or
        **bfloat16** precision, cuFFT requires transformed dimensions to satisfy
        backend-specific size restrictions.  The direct path transforms ``nlon``;
        a resampled equiangular analysis additionally
        transforms latitude length ``2 * (nlat - 1)`` for the folded midpoint operator.  The
        resampled equiangular analysis is validated for float32 and float64.  If a grid does
        not satisfy these constraints and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    coeffs = sht(signal.float())

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
    """

    def __init__(
        self,
        nlat,
        nlon,
        lmax=None,
        mmax=None,
        grid="equiangular",
        norm="ortho",
        csphase=True,
        precompute_resampling=False,
    ):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.precompute_resampling = precompute_resampling

        # Resolve once, preserving upstream triangular truncation and defaults.
        self.lmax, self.mmax = _resolve_sht_limits(nlat, nlon, lmax, mmax, grid)
        resample_latitudes = grid == "equiangular" and self.lmax > (nlat + 1) // 2
        self._resample_latitudes = resample_latitudes

        # TODO: include assertions regarding the dimensions

        # compute quadrature points and lmax based on the exactness of the quadrature
        if resample_latitudes:
            cost, weights = clenshaw_curtiss_weights(2 * nlat - 1, -1, 1)
        elif self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # fold the 2*pi longitudinal scale factor of the forward-normalized FFT into the
        # quadrature weights. It is a constant prefactor of a linear transform, so folding it
        # here is exact and saves a pointwise multiply on a complex tensor in every forward.
        weights = 2.0 * torch.pi * weights

        if resample_latitudes:
            projection_weights = _precompute_legpoly(self.mmax, self.lmax, tq[::2], norm=self.norm, csphase=self.csphase)
            quadrature_weights = weights[::2].contiguous()
            midpoint_weights = weights[1::2].contiguous()

            periodic_length = 2 * (nlat - 1)
            frequencies = torch.fft.fftfreq(periodic_length, dtype=torch.float64)
            phase = torch.polar(torch.ones_like(frequencies), torch.pi * frequencies)
            signs = torch.ones(self.mmax, 1, dtype=torch.int8)
            signs[1::2] = -1

            if self.precompute_resampling:
                effective_weights = _precompute_resampled_projection(
                    projection_weights,
                    signs,
                    quadrature_weights,
                    midpoint_weights,
                    phase,
                )
                self.register_buffer("weights", torch.view_as_real(effective_weights).contiguous(), persistent=False)
            else:
                self.register_buffer("weights", projection_weights.contiguous(), persistent=False)
                self.register_buffer("_quadrature_weights", quadrature_weights, persistent=False)
                self.register_buffer("_midpoint_weights", midpoint_weights, persistent=False)
                # Keep the phase as two real channels so module.to(dtype=...) does
                # not discard the imaginary part of a complex buffer.
                phase = torch.stack((phase.real, phase.imag), dim=0)
                self.register_buffer("_latitude_shift_phase", phase.contiguous(), persistent=False)
                self.register_buffer("_parity_signs", signs, persistent=False)
        else:
            # combine quadrature weights with the legendre weights
            pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
            weights = torch.einsum("mlk,k->mlk", pct, weights).contiguous()

            # remember quadrature weights
            self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.
        """

        check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction. The 2*pi scale factor is folded into
        # the quadrature weights, so no scaling of the complex output is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose to put the contraction dim (nlat) on the fast axis
        x = x.transpose(-1, -2)

        if self._resample_latitudes and self.precompute_resampling:
            weights = torch.view_as_complex(self.weights)
            return torch.einsum("...mk,mlk->...lm", x, weights)

        if self._resample_latitudes:
            phase = self._latitude_shift_phase.to(x.real.dtype)
            x = _fold_resampled_latitude(
                x,
                self._parity_signs,
                self._quadrature_weights.to(x.real.dtype),
                self._midpoint_weights.to(x.real.dtype),
                torch.complex(phase[0], phase[1]),
            )

        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # Legendre projection: contract over the latitude rings (stride-1 in both operands)
        w = self.weights.to(x_re.dtype)
        out_re = torch.einsum("...mk,mlk->...lm", x_re, w)
        out_im = torch.einsum("...mk,mlk->...lm", x_im, w)

        # the ...lm einsum output is non-contiguous (l ends up stride-1, m slow); torch.complex
        # preserves those strides, but inductor's meta kernel for aten.complex predicts a contiguous
        # layout, tripping assert_size_stride under torch.compile(dynamic=False). Force contiguous.
        return torch.complex(out_re.contiguous(), out_im.contiguous())


class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Given complex spherical harmonic coefficients :math:`\hat{f}_l^m`, the inverse
    scalar SHT reconstructs the real-valued signal on the sphere via Legendre
    synthesis followed by an inverse FFT:

    .. math::

        f(\theta, \lambda) = \sum_{l=0}^{l_{\max}-1} \sum_{m=0}^{m_{\max}-1}
            \hat{f}_l^m\, Y_l^m(\theta, \lambda)

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation, normalization
            conventions, grid types, and worked examples.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> isht = th.InverseRealSHT(nlat, nlon)
    >>> coeffs = torch.randn(1, isht.lmax, isht.mmax, dtype=torch.cfloat)
    >>> signal = isht(coeffs)   # shape (1, 128, 256), real
    >>> signal.shape
    torch.Size([1, 128, 256])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.irfft`) to compute the
        longitudinal inverse Fourier transform efficiently.  When running in
        **float16** or **bfloat16** precision, cuFFT requires the transformed
        dimension (``nlon``) to be a **power of two**.  If your grid does not
        satisfy this constraint and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    signal = isht(coeffs.to(torch.cfloat))

    .. note::
        The inverse real FFT (C2R transform) expects the DC component (:math:`m = 0`)
        and, when ``nlon`` is even, the Nyquist component (:math:`m = N_\lambda / 2`)
        to be purely real.  This routine zeros out the imaginary parts of these
        components before calling the transform.

    Raises
    ------
    ValueError
        If the grid type is unknown

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # Resolve once, preserving upstream triangular truncation and defaults.
        self.lmax, self.mmax = _resolve_sht_limits(nlat, nlon, lmax, mmax, grid)

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # precompute associated Legendre polynomials
        # store as (mmax, nlat, lmax) so the contraction dim l is stride-1
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = pct.permute(0, 2, 1).contiguous()

        # register buffer
        self.register_buffer("pct", pct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex spherical harmonic coefficients of shape ``(..., lmax, mmax)``.

        Returns
        -------
        torch.Tensor
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)``.
        """

        check(x.dim() >= 2, lambda: f"Expected tensor with at least 2 dimensions but got {x.dim()} instead")
        check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # legendre transformation: contract over l=lmax (stride-1 in both operands)
        # pct layout: (mmax, nlat, lmax)
        w = self.pct.to(x_re.dtype)
        out_re = torch.einsum("...ml,mkl->...km", x_re, w)
        out_im = torch.einsum("...ml,mkl->...km", x_im, w)
        # force contiguous: the einsum output is non-contiguous and inductor's aten.complex meta
        # predicts a contiguous layout, tripping assert_size_stride under torch.compile (see fwd SHT).
        x = torch.complex(out_re.contiguous(), out_im.contiguous())

        # apply the inverse (real) FFT
        x = irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x


class RealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    Decomposes a tangential vector field
    :math:`\mathbf{v} = v_\theta\,\hat{e}_\theta + v_\lambda\,\hat{e}_\lambda`
    into **spheroidal** and **toroidal** spectral coefficients
    :math:`\hat{s}_l^m` and :math:`\hat{t}_l^m` using the derivatives of the
    associated Legendre polynomials.

    On the pole-including equiangular grid, omitted limits retain the
    conservative direct-quadrature default.  When an explicit ``lmax`` exceeds
    that limit, the forward transform uses meridional resampling and folding
    for analysis beyond the direct quadrature limit before applying the
    derivative-Legendre projection, allowing recovery up to the grid's
    supported limits.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the vector SHT
            formulas, normalization conventions, and worked examples.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.
    precompute_resampling : bool
        For resampled equiangular analysis, precompute the latitude-resampling
        operator into the projection weights. This increases persistent projection
        storage and construction cost but reduces per-forward work and temporary
        memory. Has no effect when latitude resampling is not required. By default
        ``False``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> vsht = th.RealVectorSHT(nlat, nlon)
    >>> vector_field = torch.randn(1, 2, nlat, nlon)
    >>> coeffs = vsht(vector_field)   # shape (1, 2, lmax, mmax), complex
    >>> coeffs.shape
    torch.Size([1, 2, 64, 64])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.rfft`) to compute the
        longitudinal Fourier transform efficiently.  When running in **float16** or
        **bfloat16** precision, cuFFT requires transformed dimensions to satisfy
        backend-specific size restrictions.  The direct path transforms ``nlon``;
        a resampled equiangular analysis additionally
        transforms latitude length ``2 * (nlat - 1)`` for the folded midpoint operator.  The
        resampled equiangular analysis is validated for float32 and float64.  If a grid does
        not satisfy these constraints and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    coeffs = vsht(vector_field.float())

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
    """

    def __init__(
        self,
        nlat,
        nlon,
        lmax=None,
        mmax=None,
        grid="equiangular",
        norm="ortho",
        csphase=True,
        precompute_resampling=False,
    ):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.precompute_resampling = precompute_resampling

        # Resolve once, preserving upstream triangular truncation and defaults.
        self.lmax, self.mmax = _resolve_sht_limits(nlat, nlon, lmax, mmax, grid)
        resample_latitudes = grid == "equiangular" and self.lmax > (nlat + 1) // 2
        self._resample_latitudes = resample_latitudes

        # compute quadrature points
        if resample_latitudes:
            cost, weights = clenshaw_curtiss_weights(2 * nlat - 1, -1, 1)
        elif self.grid == "legendre-gauss":
            cost, weights = legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, weights = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # precompute associated Legendre polynomials. The resampled equiangular
        # path only needs the original/even latitude rings for its
        # permanent projection tensor.
        projection_tq = tq[::2] if resample_latitudes else tq
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, projection_tq, norm=self.norm, csphase=self.csphase)

        # fold the 2*pi longitudinal scale factor of the forward-normalized FFT into the
        # quadrature weights (see RealSHT.__init__)
        weights = 2.0 * torch.pi * weights

        # combine integration weights, normalization factor in to one:
        # The resampled projection is sensitive to the normalization
        # factor's precision; keep that path in the same float64 precompute dtype.
        l = torch.arange(0, self.lmax, dtype=torch.float64 if resample_latitudes else None)
        norm_factor = 1.0 / l / (l + 1)
        norm_factor[0] = 1.0
        if resample_latitudes:
            projection_weights = torch.einsum("dmlk,l->dmlk", dpct, norm_factor).contiguous()
            # since the second component is imaginary, we need to take complex conjugation into account
            projection_weights[1] = -1 * projection_weights[1]
            quadrature_weights = weights[::2].contiguous()
            midpoint_weights = weights[1::2].contiguous()

            periodic_length = 2 * (nlat - 1)
            frequencies = torch.fft.fftfreq(periodic_length, dtype=torch.float64)
            phase = torch.polar(torch.ones_like(frequencies), torch.pi * frequencies)
            signs = torch.ones(self.mmax, 1, dtype=torch.int8)
            signs[::2] = -1

            if self.precompute_resampling:
                effective_weights = _precompute_resampled_projection(
                    projection_weights,
                    signs,
                    quadrature_weights,
                    midpoint_weights,
                    phase,
                )
                self.register_buffer("weights", torch.view_as_real(effective_weights).contiguous(), persistent=False)
            else:
                self.register_buffer("weights", projection_weights, persistent=False)
                self.register_buffer("_quadrature_weights", quadrature_weights, persistent=False)
                self.register_buffer("_midpoint_weights", midpoint_weights, persistent=False)
                # Keep the phase as two real channels so module.to(dtype=...) does
                # not discard the imaginary part of a complex buffer.
                phase = torch.stack((phase.real, phase.imag), dim=0)
                self.register_buffer("_latitude_shift_phase", phase.contiguous(), persistent=False)
                self.register_buffer("_parity_signs", signs, persistent=False)
        else:
            weights = torch.einsum("dmlk,k,l->dmlk", dpct, weights, norm_factor).contiguous()
            # since the second component is imaginary, we need to take complex conjugation into account
            weights[1] = -1 * weights[1]

            # remember quadrature weights
            self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) vector spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.

        Returns
        -------
        torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.
        """

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.nlat, lambda: f"Expected latitudes shape[-2]=={self.nlat}, got {x.shape[-2]}")
        check(x.shape[-1] == self.nlon, lambda: f"Expected longitudes shape[-1]=={self.nlon}, got {x.shape[-1]}")

        # apply real fft in the longitudinal direction. The 2*pi scale factor is folded into
        # the quadrature weights, so no scaling of the complex output is needed here.
        x = rfft(x, nmodes=self.mmax, dim=-1, norm="forward")

        # transpose to put the contraction dim (nlat) on the fast axis
        x = x.transpose(-1, -2)

        if self._resample_latitudes and self.precompute_resampling:
            weights = torch.view_as_complex(self.weights)
            w0 = weights[0]
            w1 = weights[1]

            theta = x[..., 0, :, :]
            longitude = x[..., 1, :, :]
            spheroidal = torch.einsum("...mk,mlk->...lm", theta, w0) + 1j * torch.einsum("...mk,mlk->...lm", longitude, w1)
            toroidal = 1j * torch.einsum("...mk,mlk->...lm", theta, w1) - torch.einsum("...mk,mlk->...lm", longitude, w0)
            return torch.stack((spheroidal, toroidal), dim=-3)

        if self._resample_latitudes:
            phase = self._latitude_shift_phase.to(x.real.dtype)
            x = _fold_resampled_latitude(
                x,
                self._parity_signs,
                self._quadrature_weights.to(x.real.dtype),
                self._midpoint_weights.to(x.real.dtype),
                torch.complex(phase[0], phase[1]),
            )

        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        w0 = self.weights[0].to(x_re.dtype)
        w1 = self.weights[1].to(x_re.dtype)

        # contraction - spheroidal component
        s_re = torch.einsum("...mk,mlk->...lm", x_re[..., 0, :, :], w0) - torch.einsum("...mk,mlk->...lm", x_im[..., 1, :, :], w1)
        s_im = torch.einsum("...mk,mlk->...lm", x_im[..., 0, :, :], w0) + torch.einsum("...mk,mlk->...lm", x_re[..., 1, :, :], w1)

        # contraction - toroidal component
        t_re = -torch.einsum("...mk,mlk->...lm", x_im[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_re[..., 1, :, :], w0)
        t_im = torch.einsum("...mk,mlk->...lm", x_re[..., 0, :, :], w1) - torch.einsum("...mk,mlk->...lm", x_im[..., 1, :, :], w0)

        # stack the spheroidal and toroidal components in real space, so the only complex-typed
        # op is a single aten.complex over contiguous operands. Stacking complex tensors instead
        # would leave a complex cat, which inductor cannot codegen (triton has no complex type),
        # and feeding aten.complex the non-contiguous ...lm einsum outputs directly trips
        # assert_size_stride, as its meta predicts a contiguous layout. torch.stack allocates a
        # fresh contiguous buffer, which is exactly what that meta expects.
        out_re = torch.stack((s_re, t_re), dim=-3)
        out_im = torch.stack((s_im, t_im), dim=-3)

        return torch.complex(out_re, out_im)


class InverseRealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    Given spheroidal and toroidal spectral coefficients :math:`\hat{s}_l^m` and
    :math:`\hat{t}_l^m`, reconstructs the tangential vector field on the sphere
    via Legendre synthesis with the derivatives of the associated Legendre
    polynomials, followed by an inverse real FFT.

    .. seealso::
        :doc:`/guide/spherical_harmonic_transforms`
            User guide with the full mathematical derivation of the inverse
            vector SHT formulas, normalization conventions, and worked examples.

    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    lmax : int
        Maximum spherical harmonic degree
    mmax : int
        Maximum spherical harmonic order
    grid : str
        Grid type (``"equiangular"``, ``"legendre-gauss"``, ``"lobatto"``,
        ``"equiangular-trapezoidal"``), by default ``"equiangular"``
    norm : str
        Normalization convention (``"ortho"``, ``"schmidt"``, ``"unnorm"``),
        by default ``"ortho"``.
    csphase : bool
        Whether to include the Condon--Shortley phase factor :math:`(-1)^m`,
        by default ``True``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> nlat, nlon = 128, 256
    >>> ivsht = th.InverseRealVectorSHT(nlat, nlon)
    >>> coeffs = torch.randn(1, 2, ivsht.lmax, ivsht.mmax, dtype=torch.cfloat)
    >>> vector_field = ivsht(coeffs)   # shape (1, 2, 128, 256), real
    >>> vector_field.shape
    torch.Size([1, 2, 128, 256])

    .. note::
        This module uses **cuFFT** (via :func:`torch.fft.irfft`) to compute the
        longitudinal inverse Fourier transform efficiently.  When running in
        **float16** or **bfloat16** precision, cuFFT requires the transformed
        dimension (``nlon``) to be a **power of two**.  If your grid does not
        satisfy this constraint and the module is called inside a
        :class:`torch.autocast` context, guard it with
        ``torch.autocast(device_type="cuda", enabled=False)``::

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ... other half-precision work ...
                with torch.autocast(device_type="cuda", enabled=False):
                    vector_field = ivsht(coeffs.to(torch.cfloat))

    .. note::
        The inverse real FFT (C2R transform) expects the DC component (:math:`m = 0`)
        and, when ``nlon`` is even, the Nyquist component (:math:`m = N_\lambda / 2`)
        to be purely real.  This routine zeros out the imaginary parts of these
        components before calling the transform.

    References
    ----------
    :cite:`Schaeffer2013`, :cite:`Wang2018`
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # Resolve once, preserving upstream triangular truncation and defaults.
        self.lmax, self.mmax = _resolve_sht_limits(nlat, nlon, lmax, mmax, grid)

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # precompute associated Legendre polynomials
        # store as (2, mmax, nlat, lmax) so the contraction dim l is stride-1
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = dpct.permute(0, 1, 3, 2).contiguous()

        # register weights
        self.register_buffer("dpct", dpct, persistent=False)

    def extra_repr(self):
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) vector spherical harmonic transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex vector harmonic coefficients of shape ``(..., 2, lmax, mmax)``, where the
            size-2 dimension holds the spheroidal and toroidal components.

        Returns
        -------
        torch.Tensor
            Real-valued tangential vector field of shape ``(..., 2, nlat, nlon)``, where the
            size-2 dimension holds the two tangential (colatitude, longitude) components.
        """

        check(x.dim() >= 3, lambda: f"Expected tensor with at least 3 dimensions but got {x.dim()} instead")
        check(x.shape[-3] == 2, lambda: f"Expected vector field shape[-3]==2, got {x.shape[-3]}")
        check(x.shape[-2] == self.lmax, lambda: f"Expected spherical harmonic degrees (lmax) shape[-2]=={self.lmax}, got {x.shape[-2]}")
        check(x.shape[-1] == self.mmax, lambda: f"Expected spherical harmonic orders (mmax) shape[-1]=={self.mmax}, got {x.shape[-1]}")

        # transpose to put the contraction dim (lmax) on the fast axis
        x = x.transpose(-1, -2)
        x_re = x.real.contiguous()
        x_im = x.imag.contiguous()

        # dpct layout: (2, mmax, nlat, lmax) — contract over l (stride-1 in both operands)
        d0 = self.dpct[0].to(x_re.dtype)
        d1 = self.dpct[1].to(x_re.dtype)

        # contraction - spheroidal component
        srl = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d0) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d1)
        sim = torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d0) + torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d1)

        # contraction - toroidal component
        trl = -torch.einsum("...ml,mkl->...km", x_im[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_re[..., 1, :, :], d0)
        tim = torch.einsum("...ml,mkl->...km", x_re[..., 0, :, :], d1) - torch.einsum("...ml,mkl->...km", x_im[..., 1, :, :], d0)

        # reassemble in real space and apply inverse FFT, see RealVectorSHT.forward
        out_re = torch.stack((srl, trl), dim=-3)
        out_im = torch.stack((sim, tim), dim=-3)
        xs = torch.complex(out_re, out_im)
        x = irfft(xs, n=self.nlon, dim=-1, norm="forward")

        return x
