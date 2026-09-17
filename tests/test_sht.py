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

import inspect
import math
import unittest

import torch
from parameterized import parameterized, parameterized_class
from testutils import compare_tensors, disable_tf32, set_seed
from torch.autograd import gradcheck

import torch_harmonics as th
from torch_harmonics.quadrature import precompute_latitudes
from torch_harmonics.sht import (
    _fourier_shift_latitude,
    _fourier_shift_latitude_adjoint,
    _periodic_latitude_extension,
    _periodic_latitude_extension_adjoint,
)

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def random_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=False, dtype=None):
    """Random scalar SHT coefficients with proper structure:
    m=0 column real (real-valued field), triangular support (m <= l),
    and optionally l=0 row zeroed (needed when testing gradient/curl).

    ``dtype`` is the real dtype used for the coefficient components.  With no
    dtype specified, preserve the existing complex128 random draw behavior.
    """
    if dtype is None:
        c = torch.randn(batch_size, lmax, mmax, dtype=torch.complex128, device=device)
    else:
        real = torch.randn(batch_size, lmax, mmax, dtype=dtype, device=device)
        imag = torch.randn(batch_size, lmax, mmax, dtype=dtype, device=device)
        c = torch.complex(real, imag)
    c[..., :, 0] = c[..., :, 0].real
    for l in range(lmax):
        if l + 1 < mmax:
            c[..., l, l + 1 :] = 0.0
    if zero_l0:
        c[..., 0, :] = 0.0
    return c


def random_vector_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=False, dtype=torch.float64):
    """Random triangular vector SHT coefficients using the scalar helper."""
    return torch.stack(
        [
            random_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=zero_l0, dtype=dtype),
            random_sht_coeffs(batch_size, lmax, mmax, device, zero_l0=zero_l0, dtype=dtype),
        ],
        dim=-3,
    )


@parameterized_class(("device"), _devices)
class TestLegendrePolynomials(unittest.TestCase):
    """Test the associated Legendre polynomials (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()
        self.cml = lambda m, l: math.sqrt((2 * l + 1) / 4 / math.pi) * math.sqrt(math.factorial(l - m) / math.factorial(l + m))
        self.pml = dict()

        # preparing associated Legendre Polynomials (These include the Condon-Shortley phase)
        # for reference see e.g. https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        self.pml[(0, 0)] = lambda x: torch.ones_like(x)
        self.pml[(0, 1)] = lambda x: x
        self.pml[(1, 1)] = lambda x: -torch.sqrt(1.0 - x**2)
        self.pml[(0, 2)] = lambda x: 0.5 * (3 * x**2 - 1)
        self.pml[(1, 2)] = lambda x: -3 * x * torch.sqrt(1.0 - x**2)
        self.pml[(2, 2)] = lambda x: 3 * (1 - x**2)
        self.pml[(0, 3)] = lambda x: 0.5 * (5 * x**3 - 3 * x)
        self.pml[(1, 3)] = lambda x: 1.5 * (1 - 5 * x**2) * torch.sqrt(1.0 - x**2)
        self.pml[(2, 3)] = lambda x: 15 * x * (1 - x**2)
        self.pml[(3, 3)] = lambda x: -15 * torch.sqrt(1.0 - x**2) ** 3

        self.lmax = self.mmax = 4

        self.tol = 1e-9

    def test_legendre(self, verbose=False):
        if verbose:
            print(f"Testing computation of associated Legendre polynomials on {self.device.type} device")

        t = torch.linspace(0, 1, 100, dtype=torch.float64, device=self.device)
        vdm = th.legendre.legpoly(self.mmax, self.lmax, t)

        for l in range(self.lmax):
            for m in range(l + 1):
                diff = vdm[m, l] / self.cml(m, l) - self.pml[(m, l)](t)
                self.assertTrue(diff.max() <= self.tol)


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicTransform(unittest.TestCase):
    """Test the spherical harmonic transform (CPU/CUDA if available)."""

    def setUp(self):
        disable_tf32()

    def test_public_signature_and_repr(self):
        forward_expected = ["nlat", "nlon", "lmax", "mmax", "grid", "norm", "csphase", "precompute_resampling"]
        inverse_expected = ["nlat", "nlon", "lmax", "mmax", "grid", "norm", "csphase"]
        for cls in (th.RealSHT, th.RealVectorSHT):
            self.assertEqual(list(inspect.signature(cls).parameters), forward_expected)
        for cls in (th.InverseRealSHT, th.InverseRealVectorSHT):
            self.assertEqual(list(inspect.signature(cls).parameters), inverse_expected)
        self.assertFalse(inspect.signature(th.RealSHT).parameters["precompute_resampling"].default)
        self.assertFalse(inspect.signature(th.RealVectorSHT).parameters["precompute_resampling"].default)
        for cls in (th.RealSHT, th.RealVectorSHT):
            self.assertNotIn("analysis", repr(cls(9, 16, lmax=8, mmax=8)))

    def test_default_equiangular_limits_and_direct_path(self):
        with self.assertWarnsRegex(UserWarning, "Default SHT truncation"):
            default = th.RealSHT(73, 144).to(self.device)
        self.assertEqual((default.lmax, default.mmax), (37, 37))
        self.assertEqual(default.weights.shape[-1], 73)

        with self.assertWarnsRegex(UserWarning, "Default SHT truncation"):
            inverse = th.InverseRealSHT(73, 144).to(self.device)
        self.assertEqual((inverse.lmax, inverse.mmax), (37, 37))
        self.assertEqual(inverse.pct.shape[-2], 73)

        low = th.RealSHT(73, 144, lmax=20, mmax=20).to(self.device)
        low_precomputed = th.RealSHT(73, 144, lmax=20, mmax=20, precompute_resampling=True).to(self.device)
        self.assertFalse(low_precomputed._resample_latitudes)
        self.assertEqual(list(low_precomputed._buffers), ["weights"])
        set_seed(333)
        signal = torch.randn(2, 73, 144, dtype=torch.float64, device=self.device)
        self.assertTrue(
            compare_tensors(
                "default and explicit direct scalar transforms",
                default(signal)[..., :20, :20],
                low(signal),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            compare_tensors(
                "direct scalar precompute flag is harmless",
                low(signal),
                low_precomputed(signal),
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_equiangular_analysis_limits(self):
        direct = th.RealSHT(73, 144, lmax=37, mmax=37)
        self.assertEqual(direct.weights.shape[-1], 73)

        first_resampled = th.RealSHT(73, 144, lmax=38, mmax=38)
        self.assertEqual(first_resampled.weights.shape[-1], 73)
        self.assertEqual(first_resampled._midpoint_weights.shape[-1], 72)

        mmax_only = th.RealSHT(73, 144, lmax=37, mmax=72)
        self.assertEqual((mmax_only.lmax, mmax_only.mmax), (37, 37))
        self.assertEqual(mmax_only.weights.shape[-1], 73)

        full = th.RealSHT(73, 144, lmax=72, mmax=72)
        self.assertEqual((full.lmax, full.mmax), (72, 72))
        self.assertEqual(full.weights.shape[-1], 73)
        self.assertEqual(full._quadrature_weights.shape[-1], 73)
        self.assertEqual(full._midpoint_weights.shape[-1], 72)
        self.assertEqual(full._latitude_shift_phase.shape, (2, 144))

        with self.assertRaisesRegex(ValueError, r"lmax <= 72.*mmax <= 72"):
            th.RealSHT(73, 144, lmax=73, mmax=73)

        # A discarded raw limit is not an error after upstream triangular truncation.
        discarded = th.RealSHT(73, 144, lmax=100, mmax=72)
        self.assertEqual((discarded.lmax, discarded.mmax), (72, 72))

        non_equiangular = th.RealSHT(17, 32, lmax=16, mmax=16, grid="legendre-gauss")
        self.assertEqual(non_equiangular.weights.shape[-1], 17)

    def test_forward_inverse_limits_are_consistent_when_mmax_is_omitted(self):
        for nlat, nlon, expected in [(129, 144, (72, 72)), (129, 145, (73, 73))]:
            with self.subTest(nlat=nlat, nlon=nlon):
                forward = th.RealSHT(nlat, nlon, lmax=100)
                inverse = th.InverseRealSHT(nlat, nlon, lmax=100)
                self.assertEqual((forward.lmax, forward.mmax), expected)
                self.assertEqual((inverse.lmax, inverse.mmax), expected)
                self.assertEqual(inverse.pct.shape[-2], nlat)

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_equiangular_representative_modes(self, dtype):
        nlat, nlon, lmax = 73, 144, 72
        modes = [(70, 0), (70, 1), (70, 2), (70, 69), (70, 70), (71, 0), (71, 1), (71, 70), (71, 71)]
        coeffs = torch.zeros(
            len(modes),
            lmax,
            lmax,
            dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            device=self.device,
        )
        for index, (degree, order) in enumerate(modes):
            coeffs[index, degree, order] = 1.0 if order == 0 else 0.375 + 0.625j

        atol = 5e-6 if dtype == torch.float32 else 1e-10
        rtol = 5e-5 if dtype == torch.float32 else 1e-10
        inverse = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax).to(device=self.device, dtype=dtype)
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                forward = th.RealSHT(
                    nlat,
                    nlon,
                    lmax=lmax,
                    mmax=lmax,
                    precompute_resampling=precompute_resampling,
                ).to(device=self.device, dtype=dtype)
                with torch.no_grad():
                    recovered = forward(inverse(coeffs))
                self.assertTrue(compare_tensors("scalar equiangular representative modes", recovered, coeffs, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            [torch.float32, 71],
            [torch.float32, 72],
            [torch.float64, 71],
            [torch.float64, 72],
        ]
    )
    def test_equiangular_random_triangular_spectra(self, dtype, limit):
        set_seed(333)
        coeffs = random_sht_coeffs(2, limit, limit, self.device, dtype=dtype)
        atol = 5e-6 if dtype == torch.float32 else 1e-10
        rtol = 5e-5 if dtype == torch.float32 else 1e-10
        inverse = th.InverseRealSHT(73, 144, lmax=limit, mmax=limit).to(device=self.device, dtype=dtype)
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                forward = th.RealSHT(
                    73,
                    144,
                    lmax=limit,
                    mmax=limit,
                    precompute_resampling=precompute_resampling,
                ).to(device=self.device, dtype=dtype)
                with torch.no_grad():
                    recovered = forward(inverse(coeffs))
                self.assertTrue(compare_tensors("scalar equiangular random triangular spectra", recovered, coeffs, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            ["ortho", True],
            ["ortho", False],
            ["four-pi", True],
            ["four-pi", False],
            ["schmidt", True],
            ["schmidt", False],
        ]
    )
    def test_equiangular_norm_and_phase_conventions(self, norm, csphase):
        lmax = mmax = 16
        coeffs = random_sht_coeffs(2, lmax, mmax, self.device, dtype=torch.float64)
        inverse = th.InverseRealSHT(17, 32, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase).to(self.device)
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                forward = th.RealSHT(
                    17,
                    32,
                    lmax=lmax,
                    mmax=mmax,
                    norm=norm,
                    csphase=csphase,
                    precompute_resampling=precompute_resampling,
                ).to(self.device)
                with torch.no_grad():
                    recovered = forward(inverse(coeffs))
                self.assertTrue(compare_tensors("scalar equiangular norm and csphase", recovered, coeffs, atol=1e-10, rtol=1e-10))

    def test_equiangular_gradcheck(self):
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                transform = th.RealSHT(6, 12, lmax=5, mmax=5, precompute_resampling=precompute_resampling).to(self.device).double()
                signal = torch.randn(1, 6, 12, dtype=torch.float64, device=self.device, requires_grad=True)

                def loss(x):
                    coeffs = transform(x)
                    return (coeffs.real.square() + coeffs.imag.square()).sum()

                self.assertTrue(gradcheck(loss, (signal,), eps=1e-6, atol=1e-8, rtol=1e-6))

    @parameterized.expand(
        [
            # even-even
            [32, 64, 32, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "lobatto", 1e-9, 1e-9],
            # odd-even
            [33, 64, 32, "ortho", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "ortho", "lobatto", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "four-pi", "lobatto", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "equiangular", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [33, 64, 32, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_forward_inverse(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization on {self.device.type} device")

        # set seed
        set_seed(333)

        testiters = [1, 2, 4, 8, 16]
        if grid == "equiangular":
            mmax = nlat // 2
        elif grid == "lobatto":
            mmax = nlat - 1
        else:
            mmax = nlat
        lmax = mmax

        sht = th.RealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)

        # testing error accumulation
        for iter in testiters:
            with self.subTest(i=iter):
                if verbose:
                    print(f"{iter} iterations of batchsize {batch_size}:")

                base = signal

                for _ in range(iter):
                    base = isht(sht(base))

                self.assertTrue(compare_tensors(f"output iteration {iter}", base, signal, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # even-even
            [12, 24, 2, "ortho", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "ortho", "lobatto", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "four-pi", "lobatto", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "equiangular", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, 2, "schmidt", "lobatto", 1e-5, 1e-5],
            # odd-even
            [15, 30, 2, "ortho", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "ortho", "lobatto", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "four-pi", "lobatto", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "equiangular", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "legendre-gauss", 1e-5, 1e-5],
            [15, 30, 2, "schmidt", "lobatto", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_grads(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing gradients of real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization")

        # set seed
        set_seed(333)

        if grid == "equiangular":
            mmax = nlat // 2
        elif grid == "lobatto":
            mmax = nlat - 1
        else:
            mmax = nlat
        lmax = mmax

        sht = th.RealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, mmax=mmax, lmax=lmax, grid=grid, norm=norm).to(self.device)

        with torch.no_grad():
            coeffs = torch.zeros(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, device=self.device, dtype=torch.complex128)
            signal = isht(coeffs)

        # test the sht
        grad_input = torch.randn_like(signal, requires_grad=True)
        err_handle = lambda x: torch.mean(torch.norm(sht(x) - coeffs, p="fro", dim=(-1, -2)) / torch.norm(coeffs, p="fro", dim=(-1, -2)))
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=atol, rtol=rtol)
        self.assertTrue(test_result)

        # test the isht
        grad_input = torch.randn_like(coeffs, requires_grad=True)
        err_handle = lambda x: torch.mean(torch.norm(isht(x) - signal, p="fro", dim=(-1, -2)) / torch.norm(signal, p="fro", dim=(-1, -2)))
        test_result = gradcheck(err_handle, grad_input, eps=1e-6, atol=atol, rtol=rtol)
        self.assertTrue(test_result)

    @parameterized.expand(
        [
            [32, 64, 16, "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_cross_norm_consistency(self, nlat, nlon, batch_size, grid, atol, rtol, verbose=False):
        """The three normalizations applied to the same band-limited signal must satisfy
        known per-coefficient scaling relations:
          c_four_pi[l,m] = c_ortho[l,m] * sqrt(4*pi)
          c_schmidt[l,m] = c_ortho[l,m] * sqrt(4*pi / (2*l+1))
        This catches bugs where a norm is internally self-consistent but scaled wrongly
        relative to the standard conventions.
        """
        if verbose:
            print(f"Testing cross-norm consistency on {nlat}x{nlon} {grid} grid on {self.device.type}")

        # set seed
        set_seed(333)

        sht_ortho = th.RealSHT(nlat, nlon, grid=grid, norm="ortho").to(self.device)
        sht_four_pi = th.RealSHT(nlat, nlon, grid=grid, norm="four-pi").to(self.device)
        sht_schmidt = th.RealSHT(nlat, nlon, grid=grid, norm="schmidt").to(self.device)
        isht_ortho = th.InverseRealSHT(nlat, nlon, grid=grid, norm="ortho").to(self.device)

        lmax = sht_ortho.lmax
        mmax = sht_ortho.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            signal = isht_ortho(c)  # band-limited real spatial field

            c_ortho = sht_ortho(signal)
            c_four_pi = sht_four_pi(signal)
            c_schmidt = sht_schmidt(signal)

        # four-pi: each coefficient is sqrt(4*pi) larger than ortho
        c_four_pi_ref = c_ortho * math.sqrt(4.0 * math.pi)
        self.assertTrue(compare_tensors("four-pi vs ortho scaling", c_four_pi, c_four_pi_ref, atol=atol, rtol=rtol, verbose=verbose))

        # schmidt: coefficient at degree l is sqrt(4*pi/(2*l+1)) * c_ortho[l,m]
        l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
        schmidt_scale = torch.sqrt(4.0 * math.pi / (2.0 * l_vals + 1.0))  # (lmax,)
        c_schmidt_ref = c_ortho * schmidt_scale[:, None]  # broadcast over mmax
        self.assertTrue(compare_tensors("schmidt vs ortho scaling", c_schmidt, c_schmidt_ref, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_known_function(self, nlat, nlon, norm, grid, atol, rtol, verbose=False):
        """The SHT of analytically known functions must produce exact spectral coefficients.

        f(θ,φ) = 1:
          ortho   → c[0,0] = sqrt(4*pi),  all others = 0
          four-pi → c[0,0] = 4*pi,        all others = 0
          schmidt → c[0,0] = 4*pi,        all others = 0

        f(θ,φ) = cos θ:
          ortho   → c[1,0] = sqrt(4*pi/3),  all others = 0
          four-pi → c[1,0] = 4*pi/sqrt(3),  all others = 0
          schmidt → c[1,0] = 4*pi/3,        all others = 0

        These are strict value checks that will fail even when the forward/inverse
        transforms are mutually consistent but carry a wrong overall scale.
        """
        if verbose:
            print(f"Testing known-function SHT on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        sht = th.RealSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax = sht.lmax
        mmax = sht.mmax

        # colatitude nodes θ in [0, π] (north pole to south pole)
        lats, _ = precompute_latitudes(nlat, grid=grid)
        lats = lats.to(device=self.device, dtype=torch.float64)
        cost = torch.cos(lats)  # shape (nlat,)

        # ---- f = 1 (constant field) ----
        f_const = torch.ones(nlat, nlon, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            c_const = sht(f_const)  # (lmax, mmax), complex

        if norm == "ortho":
            c00_ref = math.sqrt(4.0 * math.pi)
        else:  # four-pi and schmidt agree for l=0
            c00_ref = 4.0 * math.pi

        self.assertTrue(
            compare_tensors(
                "f=1: c[0,0]",
                c_const[0:1, 0:1].real,
                torch.tensor([[c00_ref]], dtype=torch.float64, device=self.device),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        c_rest = c_const.clone()
        c_rest[0, 0] = 0.0
        self.assertTrue(
            compare_tensors(
                "f=1: all other coeffs vanish",
                c_rest.abs(),
                torch.zeros_like(c_rest.abs()),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

        # ---- f = cos θ ----
        f_costheta = cost.unsqueeze(-1).expand(nlat, nlon).contiguous()
        with torch.no_grad():
            c_cos = sht(f_costheta)  # (lmax, mmax), complex

        if norm == "ortho":
            c10_ref = math.sqrt(4.0 * math.pi / 3.0)
        elif norm == "four-pi":
            c10_ref = 4.0 * math.pi / math.sqrt(3.0)
        else:  # schmidt
            c10_ref = 4.0 * math.pi / 3.0

        self.assertTrue(
            compare_tensors(
                "f=cosθ: c[1,0]",
                c_cos[1:2, 0:1].real,
                torch.tensor([[c10_ref]], dtype=torch.float64, device=self.device),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        c_rest = c_cos.clone()
        c_rest[1, 0] = 0.0
        self.assertTrue(
            compare_tensors(
                "f=cosθ: all other coeffs vanish",
                c_rest.abs(),
                torch.zeros_like(c_rest.abs()),
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

        # ---- batched: both signals together ----
        # Exercises the leading batch dimensions (..., nlat, nlon) that the
        # single-sample checks above never reach.
        f_batch = torch.stack([f_const, f_costheta], dim=0)  # (2, nlat, nlon)
        with torch.no_grad():
            c_batch = sht(f_batch)  # (2, lmax, mmax)

        self.assertTrue(
            compare_tensors(
                "batch f=1 matches unbatched",
                c_batch[0],
                c_const,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "batch f=cosθ matches unbatched",
                c_batch[1],
                c_cos,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_csphase(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """Toggling csphase only flips the sign of odd-m spectral columns.

        csphase=True (default) multiplies the Legendre weight rows at odd m by -1.
        Applied to the same spatial signal, the two transforms must satisfy:

          sht(csphase=True) [l, m] = -sht(csphase=False)[l, m]  for odd  m
          sht(csphase=True) [l, m] =  sht(csphase=False)[l, m]  for even m

        This catches a bug where csphase is applied in both the forward and
        inverse transforms, causing the signs to cancel and hiding the error.
        """
        if verbose:
            print(f"Testing csphase sign flip on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # set seed
        set_seed(333)

        sht_cs = th.RealSHT(nlat, nlon, grid=grid, norm=norm, csphase=True).to(self.device)
        sht_no_cs = th.RealSHT(nlat, nlon, grid=grid, norm=norm, csphase=False).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = sht_cs.lmax, sht_cs.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            signal = isht(c)  # band-limited spatial field

            c_cs = sht_cs(signal)
            c_no_cs = sht_no_cs(signal)

        # build the expected sign pattern: +1 for even m, -1 for odd m
        sign = torch.ones(mmax, dtype=torch.float64, device=self.device)
        sign[1::2] = -1.0

        self.assertTrue(
            compare_tensors(
                "csphase sign flip",
                c_cs,
                c_no_cs * sign,
                atol=atol,
                rtol=rtol,
                verbose=verbose,
            )
        )

    @parameterized.expand(
        [
            [32, 64, 32, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 32, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_parseval(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """Parseval's theorem: the spatial L2 norm of isht(c) equals a weighted spectral norm
        of c.  The spectral weights W_{l,m} depend on the normalization convention:

          ortho:   W_{l,m} = w_m                        (w_m = 1 for m=0, 2 for m>0)
          four-pi: W_{l,m} = w_m / (4*pi)               (coefficients are sqrt(4*pi) larger)
          schmidt: W_{l,m} = w_m * (2*l+1) / (4*pi)     (coefficients are sqrt(4*pi/(2*l+1)) larger)

        In all cases: ||f||^2_{S^2} = sum_{l,m} W_{l,m} * |c_{l,m}|^2
        """
        if verbose:
            print(f"Testing Parseval's theorem on {nlat}x{nlon} {grid} grid with {norm} normalization on {self.device.type}")

        # set seed
        set_seed(333)

        isht = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax = isht.lmax
        mmax = isht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device)
            f = isht(c)  # (batch, nlat, nlon)

        # Spatial L2 norm via spherical quadrature: integral of f^2 over S^2
        _, w_lat = precompute_latitudes(nlat, grid=grid)
        w_lat = w_lat.to(device=self.device, dtype=torch.float64)
        dlon = 2.0 * math.pi / nlon
        spatial_norm_sq = torch.einsum("bnl,n->b", f**2, w_lat) * dlon  # (batch,)

        # Build the (lmax, mmax) spectral weight matrix W_{l,m}.
        # w_m accounts for the ±m folding in the real irfft (m=0: weight 1, m>0: weight 2).
        w_m = torch.ones(mmax, dtype=torch.float64, device=self.device)
        w_m[1:] = 2.0

        if norm == "ortho":
            # c_lm^ortho are the orthonormal coefficients; W_{l,m} = w_m
            W = w_m.unsqueeze(0).expand(lmax, mmax)
        elif norm == "four-pi":
            # c_lm^{four-pi} = sqrt(4*pi) * c_lm^ortho  =>  W_{l,m} = w_m / (4*pi)
            W = w_m.unsqueeze(0).expand(lmax, mmax) / (4.0 * math.pi)
        elif norm == "schmidt":
            # c_lm^{schmidt} = sqrt(4*pi / (2*l+1)) * c_lm^ortho  =>  W_{l,m} = w_m * (2*l+1) / (4*pi)
            l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
            W = torch.outer(2.0 * l_vals + 1.0, w_m) / (4.0 * math.pi)

        spectral_norm_sq = torch.einsum("blm,lm->b", c.abs() ** 2, W)  # (batch,)

        self.assertTrue(compare_tensors("Parseval's theorem", spatial_norm_sq, spectral_norm_sq, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # even-even
            [12, 24, "ortho", "equiangular", 1e-5, 1e-5],
            [12, 24, "ortho", "legendre-gauss", 1e-5, 1e-5],
            [12, 24, "ortho", "lobatto", 1e-5, 1e-5],
        ],
        skip_on_empty=True,
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_device_instantiation(self, nlat, nlon, norm, grid, atol, rtol, verbose=False):
        if verbose:
            print(f"Testing device instantiation of real-valued SHT on {nlat}x{nlon} {grid} grid with {norm} normalization")

        # set seed
        set_seed(333)

        # init on cpu
        sht_host = th.RealSHT(nlat, nlon, grid=grid, norm=norm)
        isht_host = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm)

        # init on device
        with torch.device(self.device):
            sht_device = th.RealSHT(nlat, nlon, grid=grid, norm=norm)
            isht_device = th.InverseRealSHT(nlat, nlon, grid=grid, norm=norm)

        self.assertTrue(compare_tensors("sht weights", sht_host.weights.cpu(), sht_device.weights.cpu(), atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("isht weights", isht_host.pct.cpu(), isht_device.pct.cpu(), atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [12, 24, 2, "equiangular", None, None],
            [12, 24, 2, "legendre-gauss", None, None],
            [11, 22, 2, "equiangular", None, None],
            [9, 16, 1, "equiangular", 8, 8],
        ],
        skip_on_empty=True,
    )
    def test_compile(self, nlat, nlon, batch_size, grid, lmax, mmax, verbose=False):
        """The scalar round trip compiles into a single graph and matches eager.

        The round trip is compiled as one function so the complex spectral coefficients
        are an *intermediate* buffer rather than a graph output -- that is the case
        inductor has to generate code for.  Triton has no complex type, so any pointwise
        kernel over a complex buffer fails codegen with ``KeyError: 'complex64'``; on CPU
        the C++ backend is used instead, where the exposure is a layout mismatch on
        ``aten.complex`` caught by ``assert_size_stride``.  Both are worth covering, which
        the CPU/CUDA parameterization of this class does.

        Backward is included deliberately: ``_EnsureContiguous.backward`` copies a complex
        gradient, and that copy only appears in the joint graph.
        """

        if verbose:
            print(f"Testing fullgraph compilation of real-valued SHT on {nlat}x{nlon} {grid} grid on {self.device.type}")

        set_seed(333)

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(self.device)
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(self.device)

        def fn(t):
            return isht(sht(t))

        x = torch.randn(batch_size, nlat, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        gradient = torch.randn_like(x)

        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=1e-5, rtol=1e-5, verbose=verbose))


class TestEquiangularSHTResampling(unittest.TestCase):
    """Test the private periodic continuation, half-shift, and adjoint helpers."""

    @staticmethod
    def _periodic_signal(frequencies, amplitudes, length, real_dtype):
        theta = 2.0 * math.pi * torch.arange(length, dtype=real_dtype) / length
        frequencies = torch.as_tensor(frequencies, dtype=real_dtype)
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        amplitudes = torch.as_tensor(amplitudes, dtype=complex_dtype)
        phase = torch.exp(torch.complex(torch.zeros_like(theta), theta)[None, :] * frequencies[:, None])
        return (amplitudes[:, None] * phase).sum(dim=0)

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_scalar_periodic_parity(self, real_dtype):
        nlat, mmax = 9, 12
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        x = torch.randn(2, 3, mmax, nlat, dtype=complex_dtype)
        signs = torch.ones(mmax, 1, dtype=torch.int8)
        signs[1::2] = -1
        extended = _periodic_latitude_extension(x, signs)

        self.assertEqual(extended.shape, (2, 3, mmax, 2 * (nlat - 1)))
        self.assertTrue(torch.equal(extended[..., :nlat], x))
        self.assertTrue(torch.equal(extended[..., nlat - 1], x[..., -1]))
        for m in range(mmax):
            sign = (-1) ** m
            for k in range(nlat - 2):
                torch.testing.assert_close(extended[..., m, nlat + k], sign * x[..., m, nlat - 2 - k])

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_vector_periodic_parity(self, real_dtype):
        nlat, mmax = 9, 12
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        x = torch.randn(2, 3, 2, mmax, nlat, dtype=complex_dtype)
        signs = torch.ones(mmax, 1, dtype=torch.int8)
        signs[::2] = -1
        extended = _periodic_latitude_extension(x, signs)

        self.assertEqual(extended.shape, (2, 3, 2, mmax, 2 * (nlat - 1)))
        self.assertTrue(torch.equal(extended[..., :nlat], x))
        self.assertTrue(torch.equal(extended[..., nlat - 1], x[..., -1]))
        for m in range(mmax):
            sign = (-1) ** (m + 1)
            for k in range(nlat - 2):
                torch.testing.assert_close(extended[..., :, m, nlat + k], sign * x[..., :, m, nlat - 2 - k])

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_fourier_shift_known_modes_and_even_nyquist(self, real_dtype):
        original_length = 16
        frequencies = [-5, -2, 3, 4]
        amplitudes = [0.3 + 0.2j, -0.4 + 0.1j, 0.15 - 0.35j, 0.2 + 0.05j]
        original = self._periodic_signal(frequencies, amplitudes, original_length, real_dtype)
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        fft_frequency = torch.fft.fftfreq(original_length, dtype=real_dtype)
        phase = torch.polar(torch.ones_like(fft_frequency), torch.pi * fft_frequency).to(complex_dtype)

        shifted = _fourier_shift_latitude(original, phase)
        frequencies = torch.as_tensor(frequencies, dtype=real_dtype)
        amplitudes = torch.as_tensor(amplitudes, dtype=complex_dtype)
        shifted_theta = 2.0 * math.pi * (torch.arange(original_length, dtype=real_dtype) + 0.5) / original_length
        expected = (amplitudes[:, None] * torch.exp(torch.complex(torch.zeros_like(shifted_theta), shifted_theta)[None, :] * frequencies[:, None])).sum(dim=0)
        tolerance = 2e-5 if real_dtype == torch.float32 else 1e-12
        torch.testing.assert_close(shifted, expected, rtol=tolerance, atol=tolerance)

        amplitude = torch.as_tensor(0.25 + 0.5j, dtype=complex_dtype)
        original_index = torch.arange(original_length, dtype=real_dtype)
        original = amplitude * (1.0 - 2.0 * original_index.remainder(2))
        shifted_index = torch.arange(original_length, dtype=real_dtype)
        expected = amplitude * torch.exp(torch.complex(torch.zeros_like(shifted_index), -math.pi * (shifted_index + 0.5)))
        torch.testing.assert_close(_fourier_shift_latitude(original, phase), expected, rtol=tolerance, atol=tolerance)

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_midpoint_operator_adjoint_identity(self, real_dtype):
        """The Fourier midpoint operator and its explicit adjoint agree under <u,v>."""

        nlat, mmax = 11, 7
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        frequency = torch.fft.fftfreq(2 * (nlat - 1), dtype=real_dtype)
        phase = torch.polar(torch.ones_like(frequency), torch.pi * frequency).to(complex_dtype)

        for vector, even_sign in ((False, 1), (True, -1)):
            with self.subTest(vector=vector):
                signs = torch.full((mmax, 1), even_sign, dtype=torch.int8)
                signs[1::2] *= -1
                if vector:
                    x = torch.randn(2, 2, mmax, nlat, dtype=complex_dtype)
                    y = torch.randn(2, 2, mmax, nlat - 1, dtype=complex_dtype)
                else:
                    x = torch.randn(2, 3, mmax, nlat, dtype=complex_dtype)
                    y = torch.randn(2, 3, mmax, nlat - 1, dtype=complex_dtype)

                midpoint = _fourier_shift_latitude(_periodic_latitude_extension(x, signs), phase)[..., : nlat - 1]
                padded = torch.cat((y, torch.zeros_like(y)), dim=-1)
                adjoint = _periodic_latitude_extension_adjoint(_fourier_shift_latitude_adjoint(padded, phase), signs)

                torch.testing.assert_close(
                    torch.vdot(midpoint.reshape(-1), y.reshape(-1)),
                    torch.vdot(x.reshape(-1), adjoint.reshape(-1)),
                    rtol=2e-5 if real_dtype == torch.float32 else 1e-12,
                    atol=2e-5 if real_dtype == torch.float32 else 1e-12,
                )


@parameterized_class(("device"), _devices)
class TestPrecomputedResampling(unittest.TestCase):
    """Test the optional precomputed latitude projection representation."""

    def setUp(self):
        disable_tf32()

    @staticmethod
    def _cases():
        return (
            (th.RealSHT, (2, 9, 16), (8, 8, 9), (8, 8, 9, 2)),
            (th.RealVectorSHT, (2, 2, 9, 16), (2, 8, 8, 9), (2, 8, 8, 9, 2)),
        )

    def test_storage_and_shapes(self):
        runtime_names = {"weights", "_quadrature_weights", "_midpoint_weights", "_latitude_shift_phase", "_parity_signs"}
        for cls, _, runtime_shape, effective_shape in self._cases():
            with self.subTest(transform=cls.__name__):
                runtime = cls(9, 16, lmax=8, mmax=8).to(self.device)
                precomputed = cls(9, 16, lmax=8, mmax=8, precompute_resampling=True).to(self.device)

                self.assertEqual(set(runtime._buffers), runtime_names)
                self.assertEqual(list(precomputed._buffers), ["weights"])
                for name in runtime_names - {"weights"}:
                    self.assertNotIn(name, precomputed._buffers)

                self.assertEqual(tuple(runtime.weights.shape), runtime_shape)
                self.assertEqual(tuple(precomputed.weights.shape), effective_shape)
                effective = torch.view_as_complex(precomputed.weights)
                self.assertEqual(tuple(effective.shape), runtime_shape)
                self.assertEqual(effective.data_ptr(), precomputed.weights.data_ptr())
                self.assertTrue(precomputed.weights.is_contiguous())
                self.assertGreater(effective.imag.abs().max().item(), 0.0)

                runtime_bytes = runtime.weights.numel() * runtime.weights.element_size()
                effective_bytes = precomputed.weights.numel() * precomputed.weights.element_size()
                self.assertEqual(effective_bytes, 2 * runtime_bytes)

    def test_runtime_and_precomputed_equivalence(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.float64, 2e-12)):
            for cls, _, _, _ in self._cases():
                with self.subTest(transform=cls.__name__, dtype=dtype):
                    runtime = cls(73, 144, lmax=72, mmax=72).to(device=self.device, dtype=dtype)
                    precomputed = cls(73, 144, lmax=72, mmax=72, precompute_resampling=True).to(device=self.device, dtype=dtype)
                    set_seed(8128)
                    if cls is th.RealSHT:
                        shape = (2, 73, 144)
                    else:
                        shape = (2, 2, 73, 144)
                    x = torch.randn(*shape, device=self.device, dtype=dtype)
                    runtime_output = runtime(x)
                    precomputed_output = precomputed(x)
                    difference = (runtime_output - precomputed_output).abs()
                    self.assertLessEqual(difference.max().item(), tolerance)
                    self.assertLessEqual(difference.norm().item() / runtime_output.norm().item(), tolerance)

    def test_non_equiangular_path_ignores_flag(self):
        for cls, _, _, _ in self._cases():
            with self.subTest(transform=cls.__name__):
                runtime = cls(17, 32, lmax=16, mmax=16, grid="legendre-gauss")
                precomputed = cls(17, 32, lmax=16, mmax=16, grid="legendre-gauss", precompute_resampling=True)
                self.assertFalse(precomputed._resample_latitudes)
                self.assertEqual(list(precomputed._buffers), ["weights"])
                self.assertTrue(torch.equal(runtime.weights, precomputed.weights))

    def test_dtype_and_device_lifecycle(self):
        for cls, _, _, _ in self._cases():
            with self.subTest(transform=cls.__name__):
                transitions = [
                    ("float", lambda m: m.float()),
                    ("double", lambda m: m.double()),
                    ("to_float", lambda m: m.to(dtype=torch.float32)),
                    ("to_double", lambda m: m.to(dtype=torch.float64)),
                ]
                if torch.cuda.is_available():
                    transitions.extend(
                        [
                            ("cuda", lambda m: m.cuda()),
                            ("cpu", lambda m: m.cpu()),
                            ("float_cuda", lambda m: m.float().cuda()),
                            ("cpu_double", lambda m: m.cpu().double()),
                        ]
                    )

                for name, transition in transitions:
                    with self.subTest(transition=name):
                        module = cls(9, 16, lmax=8, mmax=8, precompute_resampling=True)
                        module = transition(module)
                        real_dtype = module.weights.dtype
                        expected_complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
                        device = module.weights.device
                        if cls is th.RealSHT:
                            shape = (2, 9, 16)
                        else:
                            shape = (2, 2, 9, 16)
                        x = torch.randn(*shape, device=device, dtype=real_dtype)
                        expected = cls(9, 16, lmax=8, mmax=8).to(device=device, dtype=real_dtype)(x)
                        actual = module(x)
                        self.assertTrue(
                            compare_tensors(
                                "lifecycle output", actual, expected, atol=2e-5 if real_dtype == torch.float32 else 2e-12, rtol=2e-5 if real_dtype == torch.float32 else 2e-12
                            )
                        )

                        complex_weights = torch.view_as_complex(module.weights)
                        self.assertEqual(complex_weights.dtype, expected_complex_dtype)
                        self.assertEqual(complex_weights.data_ptr(), module.weights.data_ptr())
                        self.assertGreater(complex_weights.imag.abs().max().item(), 0.0)

    def test_input_gradients_and_gradcheck(self):
        for cls, _, _, _ in self._cases():
            with self.subTest(transform=cls.__name__):
                runtime = cls(6, 12, lmax=5, mmax=5).to(device=self.device, dtype=torch.float64)
                precomputed = cls(6, 12, lmax=5, mmax=5, precompute_resampling=True).to(device=self.device, dtype=torch.float64)
                shape = (1, 6, 12) if cls is th.RealSHT else (1, 2, 6, 12)
                set_seed(9012)
                base = torch.randn(*shape, device=self.device, dtype=torch.float64)
                runtime_input = base.clone().requires_grad_()
                precomputed_input = base.clone().requires_grad_()

                def energy(module, value):
                    output = module(value)
                    return output.real.square().mean() + output.imag.square().mean()

                (runtime_gradient,) = torch.autograd.grad(energy(runtime, runtime_input), runtime_input)
                (precomputed_gradient,) = torch.autograd.grad(energy(precomputed, precomputed_input), precomputed_input)
                torch.testing.assert_close(precomputed_gradient, runtime_gradient, rtol=2e-10, atol=2e-10)

                def loss(value):
                    return energy(precomputed, value)

                check_input = base[:1].clone().requires_grad_()
                self.assertTrue(gradcheck(loss, (check_input,), eps=1e-6, atol=1e-8, rtol=1e-6))

    def test_compile_forward_and_backward(self):
        for cls, _, _, _ in self._cases():
            with self.subTest(transform=cls.__name__):
                module = cls(9, 16, lmax=8, mmax=8, precompute_resampling=True).to(device=self.device, dtype=torch.float32)
                shape = (2, 9, 16) if cls is th.RealSHT else (2, 2, 9, 16)
                set_seed(4567)
                x = torch.randn(*shape, device=self.device, dtype=torch.float32, requires_grad=True)

                eager_output = module(x)
                eager_loss = eager_output.real.square().mean() + eager_output.imag.square().mean()
                (eager_gradient,) = torch.autograd.grad(eager_loss, x)

                compiled = torch.compile(module, fullgraph=True, dynamic=False)
                compiled_output = compiled(x)
                compiled_loss = compiled_output.real.square().mean() + compiled_output.imag.square().mean()
                (compiled_gradient,) = torch.autograd.grad(compiled_loss, x)

                torch.testing.assert_close(compiled_output, eager_output, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(compiled_gradient, eager_gradient, rtol=1e-5, atol=1e-5)


@parameterized_class(("device"), _devices)
class TestSphericalHarmonicsFunctions(unittest.TestCase):
    """Test fundamental properties of the real spherical harmonic basis functions.

    InverseRealSHT with norm="ortho" synthesizes orthonormal basis functions on
    the sphere.  Setting a single complex coefficient c_{l,m} = 1 synthesizes

      f_{l,0}      = Y_l^0(theta, phi)                   for m = 0
      f_{l,m,cos}  ~ P_l^m(cos theta) * cos(m*phi)       for m > 0  (c_{l,m} = 1+0j)
      f_{l,m,sin}  ~ P_l^m(cos theta) * sin(m*phi)       for m > 0  (c_{l,m} = 0+1j)

    With ortho normalization the sphere inner products satisfy:
      <f_{l,0},     f_{l',0}    > = delta_{ll'}
      <f_{l,m,cos}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0
      <f_{l,m,sin}, f_{l',m',*}> = 2 * delta_{ll'} * delta_{mm'}   for m, m' > 0

    The factor of 2 for m > 0 arises because the real irfft folds the +m and -m
    modes together, doubling the amplitude of each mode.
    """

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [
            [12, 24, "legendre-gauss", 1e-9, 1e-9],
            [12, 24, "equiangular", 1e-9, 1e-9],
            [12, 24, "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_orthogonality(self, nlat, nlon, grid, atol, rtol, verbose=False):
        """Verify that isht(norm="ortho") synthesizes mutually orthogonal basis
        functions and that the self inner-products equal 1 (m=0) or 2 (m>0)."""
        if verbose:
            print(f"Testing Y_lm orthogonality on {nlat}x{nlon} {grid} grid on {self.device.type}")

        # set seed
        set_seed(333)

        if grid == "equiangular":
            lmax = mmax = nlat // 2
        elif grid == "lobatto":
            lmax = mmax = nlat - 1
        else:
            lmax = mmax = nlat

        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, norm="ortho").to(self.device)

        # Build one coefficient tensor per real basis function.
        # For m = 0: one tensor with c[l, 0] = 1+0j (real mode only).
        # For m > 0: two tensors — c[l, m] = 1+0j (cos) and c[l, m] = 0+1j (sin).
        basis_list = []
        expected_diag = []
        for l in range(lmax):
            for m in range(min(l + 1, mmax)):
                c = torch.zeros(lmax, mmax, dtype=torch.complex128)
                c[l, m] = 1.0 + 0.0j
                basis_list.append(c)
                expected_diag.append(1.0 if m == 0 else 2.0)
                if m > 0:
                    c = torch.zeros(lmax, mmax, dtype=torch.complex128)
                    c[l, m] = 0.0 + 1.0j
                    basis_list.append(c)
                    expected_diag.append(2.0)

        coeffs = torch.stack(basis_list).to(self.device)  # (N, lmax, mmax)

        with torch.no_grad():
            funcs = isht(coeffs)  # (N, nlat, nlon), real-valued

        # Gram matrix via spherical quadrature: G[i,j] = integral of f_i * f_j over S^2
        # Weights from precompute_latitudes are in the cos(theta) domain and integrate
        # over [-1, 1], so the full measure is w_lat[k] * dlon.
        _, w_lat = precompute_latitudes(nlat, grid=grid)
        w_lat = w_lat.to(device=self.device, dtype=torch.float64)
        dlon = 2.0 * math.pi / nlon

        weighted = funcs * (dlon * w_lat).unsqueeze(-1)  # (N, nlat, nlon)
        gram = torch.einsum("inl,jnl->ij", weighted, funcs)  # (N, N)

        expected = torch.diag(torch.tensor(expected_diag, dtype=torch.float64, device=self.device))
        self.assertTrue(compare_tensors("Gram matrix", gram, expected, atol=atol, rtol=rtol, verbose=verbose))


@parameterized_class(("device"), _devices)
class TestVectorSphericalHarmonicTransform(unittest.TestCase):
    """Tests for the consistency between the scalar SHT and the vector SHT.

    RealVectorSHT includes a 1/(l*(l+1)) normalization in its quadrature weights
    so that the spheroidal/toroidal spectral coefficients relate directly to the
    scalar SHT coefficients of the generating potential:

      Gradient:  vsht(ivsht([c, 0]))[spheroidal] = c,  [toroidal] = 0  (l > 0)
      Curl:      vsht(ivsht([0, c]))[spheroidal] = 0,  [toroidal] = c  (l > 0)

    The l = 0 mode is zero in both vsht and ivsht because the gradient and curl
    of a constant field (Y_0^0) vanish identically on the sphere.

    These tests catch swapped spheroidal/toroidal channels, wrong signs in the
    dP/dtheta or P/sin(theta) terms, and incorrect l*(l+1) normalization.
    """

    def setUp(self):
        disable_tf32()

    def test_default_equiangular_limits_and_direct_path(self):
        with self.assertWarnsRegex(UserWarning, "Default SHT truncation"):
            default = th.RealVectorSHT(73, 144).to(self.device)
        self.assertEqual((default.lmax, default.mmax), (37, 37))
        self.assertEqual(default.weights.shape[-1], 73)

        with self.assertWarnsRegex(UserWarning, "Default SHT truncation"):
            inverse = th.InverseRealVectorSHT(73, 144).to(self.device)
        self.assertEqual((inverse.lmax, inverse.mmax), (37, 37))
        self.assertEqual(inverse.dpct.shape[-2], 73)

        low = th.RealVectorSHT(73, 144, lmax=20, mmax=20).to(self.device)
        low_precomputed = th.RealVectorSHT(73, 144, lmax=20, mmax=20, precompute_resampling=True).to(self.device)
        self.assertFalse(low_precomputed._resample_latitudes)
        self.assertEqual(list(low_precomputed._buffers), ["weights"])
        set_seed(333)
        vector_field = torch.randn(2, 2, 73, 144, dtype=torch.float64, device=self.device)
        self.assertTrue(
            compare_tensors(
                "default and explicit direct vector transforms",
                default(vector_field)[..., :20, :20],
                low(vector_field),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            compare_tensors(
                "direct vector precompute flag is harmless",
                low(vector_field),
                low_precomputed(vector_field),
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_equiangular_analysis_limits(self):
        direct = th.RealVectorSHT(73, 144, lmax=37, mmax=37)
        self.assertEqual(direct.weights.shape[-1], 73)

        first_resampled = th.RealVectorSHT(73, 144, lmax=38, mmax=38)
        self.assertEqual(first_resampled.weights.shape[-1], 73)
        self.assertEqual(first_resampled._midpoint_weights.shape[-1], 72)

        mmax_only = th.RealVectorSHT(73, 144, lmax=37, mmax=72)
        self.assertEqual((mmax_only.lmax, mmax_only.mmax), (37, 37))
        self.assertEqual(mmax_only.weights.shape[-1], 73)

        full = th.RealVectorSHT(73, 144, lmax=72, mmax=72)
        self.assertEqual((full.lmax, full.mmax), (72, 72))
        self.assertEqual(full.weights.shape[-1], 73)
        self.assertEqual(full._quadrature_weights.shape[-1], 73)
        self.assertEqual(full._midpoint_weights.shape[-1], 72)
        self.assertEqual(full._latitude_shift_phase.shape, (2, 144))

        with self.assertRaisesRegex(ValueError, r"lmax <= 72.*mmax <= 72"):
            th.RealVectorSHT(73, 144, lmax=73, mmax=73)

        # A discarded raw limit is not an error after upstream triangular truncation.
        discarded = th.RealVectorSHT(73, 144, lmax=100, mmax=72)
        self.assertEqual((discarded.lmax, discarded.mmax), (72, 72))

        non_equiangular = th.RealVectorSHT(17, 32, lmax=16, mmax=16, grid="legendre-gauss")
        self.assertEqual(non_equiangular.weights.shape[-1], 17)

    def test_forward_inverse_limits_are_consistent_when_mmax_is_omitted(self):
        for nlat, nlon, expected in [(129, 144, (72, 72)), (129, 145, (73, 73))]:
            with self.subTest(nlat=nlat, nlon=nlon):
                forward = th.RealVectorSHT(nlat, nlon, lmax=100)
                inverse = th.InverseRealVectorSHT(nlat, nlon, lmax=100)
                self.assertEqual((forward.lmax, forward.mmax), expected)
                self.assertEqual((inverse.lmax, inverse.mmax), expected)
                self.assertEqual(inverse.dpct.shape[-2], nlat)

    @parameterized.expand([[torch.float32], [torch.float64]])
    def test_equiangular_spheroidal_and_toroidal_modes(self, dtype):
        nlat, nlon, lmax = 73, 144, 72
        modes = [(70, 0), (70, 1), (70, 70), (71, 0), (71, 1), (71, 71)]
        inverse = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=lmax).to(device=self.device, dtype=dtype)
        atol = 5e-6 if dtype == torch.float32 else 1e-10
        rtol = 5e-5 if dtype == torch.float32 else 1e-10
        for precompute_resampling in (False, True):
            forward = th.RealVectorSHT(
                nlat,
                nlon,
                lmax=lmax,
                mmax=lmax,
                precompute_resampling=precompute_resampling,
            ).to(device=self.device, dtype=dtype)
            for channel in range(2):
                with self.subTest(channel=channel, precompute_resampling=precompute_resampling):
                    coeffs = torch.zeros(
                        len(modes),
                        2,
                        lmax,
                        lmax,
                        dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
                        device=self.device,
                    )
                    for index, (degree, order) in enumerate(modes):
                        value = 1.0 if order == 0 else 0.375 + 0.625j
                        coeffs[index, channel, degree, order] = value

                    with torch.no_grad():
                        recovered = forward(inverse(coeffs))

                    self.assertTrue(compare_tensors("vector equiangular representative modes", recovered, coeffs, atol=atol, rtol=rtol))

    @parameterized.expand(
        [
            [torch.float32, 71],
            [torch.float32, 72],
            [torch.float64, 71],
            [torch.float64, 72],
        ]
    )
    def test_equiangular_random_triangular_spectra(self, dtype, limit):
        set_seed(333)
        coeffs = random_vector_sht_coeffs(2, limit, limit, self.device, zero_l0=True, dtype=dtype)
        inverse = th.InverseRealVectorSHT(73, 144, lmax=limit, mmax=limit).to(device=self.device, dtype=dtype)
        if dtype == torch.float32:
            # dP/dtheta and P/sin(theta) contractions accumulate a few more
            # float32 ulps than the scalar projection at the highest degrees.
            atol, rtol = 2.5e-5, 1e-6
        else:
            atol, rtol = 1e-10, 1e-10
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                forward = th.RealVectorSHT(
                    73,
                    144,
                    lmax=limit,
                    mmax=limit,
                    precompute_resampling=precompute_resampling,
                ).to(device=self.device, dtype=dtype)
                with torch.no_grad():
                    recovered = forward(inverse(coeffs))
                difference = (recovered - coeffs).abs()
                relative_error = difference.norm() / coeffs.abs().norm()
                self.assertLessEqual(relative_error.item(), rtol)
                self.assertLessEqual(difference.max().item(), atol)

    @parameterized.expand(
        [
            ["ortho", True],
            ["ortho", False],
            ["four-pi", True],
            ["four-pi", False],
            ["schmidt", True],
            ["schmidt", False],
        ]
    )
    def test_equiangular_norm_and_phase_conventions(self, norm, csphase):
        lmax = mmax = 16
        coeffs = random_vector_sht_coeffs(2, lmax, mmax, self.device, zero_l0=True, dtype=torch.float64)
        inverse = th.InverseRealVectorSHT(17, 32, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase).to(self.device)
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                forward = th.RealVectorSHT(
                    17,
                    32,
                    lmax=lmax,
                    mmax=mmax,
                    norm=norm,
                    csphase=csphase,
                    precompute_resampling=precompute_resampling,
                ).to(self.device)
                with torch.no_grad():
                    recovered = forward(inverse(coeffs))
                self.assertTrue(compare_tensors("vector equiangular norm and csphase", recovered, coeffs, atol=1e-10, rtol=1e-10))

    def test_equiangular_backward(self):
        for precompute_resampling in (False, True):
            with self.subTest(precompute_resampling=precompute_resampling):
                transform = th.RealVectorSHT(8, 16, lmax=7, mmax=7, precompute_resampling=precompute_resampling).to(self.device).double()
                vector_field = torch.randn(2, 2, 8, 16, dtype=torch.float64, device=self.device, requires_grad=True)
                transform(vector_field).abs().square().mean().backward()
                self.assertIsNotNone(vector_field.grad)
                self.assertTrue(torch.isfinite(vector_field.grad).all())

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "lobatto", 1e-7, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_gradient_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=True):
        """ivsht([c, 0]) synthesizes the surface gradient ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this gradient field must recover c in the
        spheroidal channel and zero in the toroidal channel, because a gradient
        field is curl-free (purely spheroidal).
        """
        if verbose:
            print(f"Testing gradient consistency on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # set seed
        set_seed(333)

        vsht = th.RealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize gradient field: ivsht([c, 0]) = ∇_S f
            grad_f = ivsht(torch.stack([c, zeros], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(∇_S f) must give [c, 0]
            st = vsht(grad_f)  # (batch, 2, lmax, mmax)
            s = st[..., 0, :, :]  # spheroidal
            t = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, c, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients", t, zeros, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "ortho", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "four-pi", "lobatto", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "equiangular", 1e-7, 1e-7],
            [32, 64, 16, "schmidt", "lobatto", 1e-7, 1e-7],
        ],
        skip_on_empty=True,
    )
    def test_curl_consistency(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """ivsht([0, c]) synthesizes the surface curl ê_r × ∇_S f of a scalar field
        f = isht(c).  Applying vsht to this curl field must recover c in the
        toroidal channel and zero in the spheroidal channel, because a surface
        curl field is divergence-free (purely toroidal).
        """
        if verbose:
            print(f"Testing curl consistency on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # set seed
        set_seed(333)

        vsht = th.RealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        with torch.no_grad():
            c = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            zeros = torch.zeros_like(c)

            # synthesize curl field: ivsht([0, c]) = ê_r × ∇_S f
            curl_f = ivsht(torch.stack([zeros, c], dim=-3))  # (batch, 2, nlat, nlon)

            # analyse: vsht(ê_r × ∇_S f) must give [0, c]
            st = vsht(curl_f)  # (batch, 2, lmax, mmax)
            s = st[..., 0, :, :]  # spheroidal
            t = st[..., 1, :, :]  # toroidal

        self.assertTrue(compare_tensors("spheroidal coefficients", s, zeros, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(compare_tensors("toroidal coefficients", t, c, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            # The spatial round-trip ivsht(vsht(v)) ≈ v is limited to ~1e-5 accuracy
            # because dP/dθ and P/sinθ are not polynomials in cos θ, so Gauss quadrature
            # cannot integrate their products exactly (unlike the scalar SHT).
            [32, 64, 16, "ortho", "legendre-gauss", 5e-4, 1e-4],
            [32, 64, 16, "ortho", "equiangular", 5e-4, 1e-4],
            [32, 64, 16, "ortho", "lobatto", 5e-4, 1e-4],
            [32, 64, 16, "four-pi", "legendre-gauss", 5e-4, 1e-4],
            [32, 64, 16, "four-pi", "equiangular", 5e-4, 1e-4],
            [32, 64, 16, "four-pi", "lobatto", 5e-4, 1e-4],
            [32, 64, 16, "schmidt", "legendre-gauss", 5e-4, 5e-4],
            [32, 64, 16, "schmidt", "equiangular", 5e-4, 5e-4],
            [32, 64, 16, "schmidt", "lobatto", 5e-4, 5e-4],
        ],
        skip_on_empty=True,
    )
    def test_vector_forward_inverse(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """ivsht(vsht(v)) ≈ v for a band-limited spatial vector field.

        Unlike the gradient/curl consistency tests — which start in spectral space
        with single-channel inputs — this test starts in spatial space with a general
        two-channel vector field and verifies that vsht and ivsht are genuine left-
        inverses of each other across multiple iterations.
        """
        if verbose:
            print(f"Testing vector SHT forward-inverse on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # set seed
        set_seed(333)

        vsht = th.RealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        ivsht = th.InverseRealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = vsht.lmax, vsht.mmax

        testiters = [1, 2, 4, 8, 16]

        with torch.no_grad():
            c_s = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            c_t = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            v = ivsht(torch.stack([c_s, c_t], dim=-3))  # (batch, 2, nlat, nlon)

        for iter in testiters:
            with self.subTest(i=iter):
                base = v
                for _ in range(iter):
                    base = ivsht(vsht(base))
                self.assertTrue(compare_tensors(f"vector round-trip iter {iter}", base, v, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [32, 64, 16, "ortho", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "ortho", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "four-pi", "lobatto", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "legendre-gauss", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "equiangular", 1e-9, 1e-9],
            [32, 64, 16, "schmidt", "lobatto", 1e-9, 1e-9],
        ],
        skip_on_empty=True,
    )
    def test_vector_parseval(self, nlat, nlon, batch_size, norm, grid, atol, rtol, verbose=False):
        """The spatial L2 energy of a vector field equals a weighted spectral sum.

        For a tangent vector field v = ivsht([c_s, c_t]), the energy identity is:

          ∫_{S²} (v_θ² + v_φ²) dΩ  =  sum_{l,m} W[l,m] * (|c_s_{l,m}|² + |c_t_{l,m}|²)

        The spectral weights W[l,m] (with w_m = 1 for m=0, 2 for m>0) are:

          ortho:   W[l,m] = w_m * l*(l+1)
          four-pi: W[l,m] = w_m * l*(l+1) / (4*pi)
          schmidt: W[l,m] = w_m * l*(l+1) * (2*l+1) / (4*pi)

        The extra l*(l+1) factor relative to scalar Parseval comes from
        ||∇_S Y_l^m||² = l*(l+1).

        The spectral energy is computed from the synthesis coefficients directly
        (not via vsht) so the test is exact to quadrature precision, mirroring
        the scalar Parseval test which uses c rather than sht(f).
        """
        if verbose:
            print(f"Testing vector Parseval's theorem on {nlat}x{nlon} {grid} grid with {norm} norm on {self.device.type}")

        # set seed
        set_seed(333)

        ivsht = th.InverseRealVectorSHT(nlat, nlon, grid=grid, norm=norm).to(self.device)
        lmax, mmax = ivsht.lmax, ivsht.mmax

        with torch.no_grad():
            c_s = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            c_t = random_sht_coeffs(batch_size, lmax, mmax, self.device, zero_l0=True)
            v = ivsht(torch.stack([c_s, c_t], dim=-3))  # (batch, 2, nlat, nlon)

        # Spatial L2 energy via spherical quadrature over both vector components
        _, w_lat = precompute_latitudes(nlat, grid=grid)
        w_lat = w_lat.to(device=self.device, dtype=torch.float64)
        dlon = 2.0 * math.pi / nlon
        spatial_energy = torch.einsum("bvnl,n->b", v**2, w_lat) * dlon  # (batch,)

        # Build spectral weight matrix W[l,m]
        w_m = torch.ones(mmax, dtype=torch.float64, device=self.device)
        w_m[1:] = 2.0
        l_vals = torch.arange(lmax, dtype=torch.float64, device=self.device)
        ll1 = l_vals * (l_vals + 1.0)  # l*(l+1); zero at l=0, matching zero vector energy there

        if norm == "ortho":
            W = torch.outer(ll1, w_m)
        elif norm == "four-pi":
            W = torch.outer(ll1, w_m) / (4.0 * math.pi)
        elif norm == "schmidt":
            W = torch.outer(ll1 * (2.0 * l_vals + 1.0), w_m) / (4.0 * math.pi)

        spectral_energy = torch.einsum("blm,lm->b", c_s.abs() ** 2 + c_t.abs() ** 2, W)  # (batch,)

        self.assertTrue(compare_tensors("vector Parseval's theorem", spatial_energy, spectral_energy, atol=atol, rtol=rtol, verbose=verbose))

    @parameterized.expand(
        [
            [12, 24, 2, "equiangular", None, None],
            [12, 24, 2, "legendre-gauss", None, None],
            [11, 22, 2, "equiangular", None, None],
            [9, 16, 1, "equiangular", 8, 8],
        ],
        skip_on_empty=True,
    )
    def test_compile(self, nlat, nlon, batch_size, grid, lmax, mmax, verbose=False):
        """The vector round trip compiles into a single graph and matches eager.

        Same rationale as the scalar case, see
        ``TestSphericalHarmonicTransform.test_compile``.  The vector transforms assemble
        their spheroidal and toroidal components separately, so they exercise a code path
        the scalar test does not reach.
        """

        if verbose:
            print(f"Testing fullgraph compilation of vector SHT on {nlat}x{nlon} {grid} grid on {self.device.type}")

        set_seed(333)

        vsht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(self.device)
        ivsht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(self.device)

        def fn(t):
            return ivsht(vsht(t))

        x = torch.randn(batch_size, 2, nlat, nlon, device=self.device, dtype=torch.float32, requires_grad=True)
        gradient = torch.randn_like(x)

        expected = fn(x)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_outputs=gradient)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(x)
        (actual_grad,) = torch.autograd.grad(actual, x, grad_outputs=gradient)

        self.assertTrue(compare_tensors("compiled forward", actual, expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        self.assertTrue(compare_tensors("compiled backward", actual_grad, expected_grad, atol=1e-5, rtol=1e-5, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
