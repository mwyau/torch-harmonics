# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
from parameterized import parameterized
from testutils import compare_tensors

import torch_harmonics as th
from torch_harmonics.truncation import SHTTruncation


def spectral_support(lmax, mmax, lmmax=None):
    m = torch.arange(mmax).view(-1, 1)
    l = torch.arange(lmax).view(1, -1)
    return (m <= l) if lmmax is None else (m <= l) & (l - m < lmmax)


class TestSHTTruncation(unittest.TestCase):
    def test_grid_defaults(self):
        for grid, expected_lmax in (("legendre-gauss", 32), ("lobatto", 31), ("equiangular", 16), ("equiangular-trapezoidal", 16)):
            with self.subTest(grid=grid):
                self.assertEqual(th.truncate_sht(32, 16, grid=grid), SHTTruncation(expected_lmax, 9, None))
                self.assertEqual(th.truncate_sht(32, 64, grid=grid), SHTTruncation(expected_lmax, expected_lmax, None))
                self.assertEqual(th.truncate_sht(32, 64, grid=grid, lmmax=5), SHTTruncation(expected_lmax, expected_lmax, 5))

    def test_independent_limits(self):
        cases = (
            ((32, 16), {}, (32, 9, None)),
            ((32, 16), {"lmax": 24}, (24, 9, None)),
            ((32, 16), {"mmax": 12}, (32, 12, None)),
            ((16, 32), {"lmax": 12, "mmax": 5}, (12, 5, None)),
            ((16, 32), {"lmax": 5, "mmax": 12}, (5, 5, None)),
            ((16, 32), {"lmax": 9, "mmax": 5, "lmmax": 5}, (9, 5, 5)),
            ((16, 32), {"lmax": 7, "mmax": 5, "lmmax": 5}, (7, 5, 5)),
            ((16, 32), {"lmax": 12, "lmmax": 5}, (12, 12, 5)),
            ((16, 32), {"mmax": 5, "lmmax": 5}, (16, 5, 5)),
            ((16, 32), {"lmax": 5, "mmax": 5, "lmmax": 1}, (5, 5, 1)),
            ((16, 32), {"lmax": 5, "mmax": 3, "lmmax": 20}, (5, 3, 20)),
        )
        for (nlat, nlon), limits, expected in cases:
            with self.subTest(nlat=nlat, nlon=nlon, limits=limits):
                self.assertEqual(th.truncate_sht(nlat, nlon, grid="legendre-gauss", **limits), SHTTruncation(*expected))

    def test_descriptor_and_positional_arguments(self):
        trunc = th.truncate_sht(16, 32, 9, 5, "legendre-gauss", lmmax=5)
        self.assertIsInstance(trunc, SHTTruncation)
        with self.assertRaises(AttributeError):
            trunc.lmmax = 4
        for cls in (th.RealSHT, th.InverseRealSHT, th.RealVectorSHT, th.InverseRealVectorSHT):
            transform = cls(16, 32, 9, 5, "legendre-gauss", "ortho", False, lmmax=5)
            self.assertEqual(transform._trunc, trunc)
            self.assertEqual((transform.lmax, transform.mmax, transform.lmmax), trunc)
            self.assertFalse(transform.csphase)
            self.assertIn("lmmax=5", repr(transform))
            for name in ("lmax", "mmax", "lmmax"):
                self.assertNotIn(name, vars(transform))
                with self.assertRaises(AttributeError):
                    setattr(transform, name, 2)

    def test_invalid_limits(self):
        for name in ("lmax", "mmax", "lmmax"):
            for invalid in (0, -1, 1.5, True, "5"):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, name + " must be a positive integer"):
                        th.truncate_sht(16, 32, grid="legendre-gauss", **{name: invalid})
        for cls in (th.truncate_sht, th.RealSHT, th.InverseRealSHT, th.RealVectorSHT, th.InverseRealVectorSHT):
            with self.assertRaisesRegex(ValueError, "mmax must not exceed lmax when lmmax is provided"):
                cls(16, 32, lmax=5, mmax=9, grid="legendre-gauss", lmmax=9)

    @parameterized.expand([(5, 5, None), (12, 5, None), (9, 5, 5), (7, 5, 5), (5, 5, 1), (5, 3, 20)])
    def test_dense_support(self, lmax, mmax, lmmax):
        """Each order ends at min(lmax, m + lmmax), with independent limits."""
        support = spectral_support(lmax, mmax, lmmax)
        for m in range(mmax):
            upper = lmax if lmmax is None else min(lmax, m + lmmax)
            self.assertEqual(torch.where(support[m])[0].tolist(), list(range(m, upper)))
        if (lmax, mmax, lmmax) == (7, 5, 5):
            # The bandwidth ends the first two orders; the degree cap ends the last two.
            self.assertEqual([torch.where(row)[0][-1].item() + 1 for row in support], [5, 6, 7, 7, 7])
        for precompute, direct in ((th.legendre._precompute_legpoly, th.legendre.legpoly), (th.legendre._precompute_dlegpoly, th.legendre.dlegpoly)):
            trunc = th.truncate_sht(16, 32, lmax=lmax, mmax=mmax, grid="legendre-gauss", lmmax=lmmax)
            dense = precompute(mmax, lmax, 16, "legendre-gauss")
            actual = precompute(mmax, lmax, 16, "legendre-gauss", trunc=trunc)
            expected = dense.clone().masked_fill_(~support.unsqueeze(-1), 0.0)
            self.assertTrue(torch.equal(actual, expected))
            # Every physical retained mode has some nonzero scalar value on this grid.
            if direct is th.legendre.legpoly:
                self.assertTrue(torch.equal(actual.ne(0).any(dim=-1), support))

    def test_r42_boundary(self):
        trunc = th.truncate_sht(96, 192, lmax=85, mmax=43, grid="legendre-gauss", lmmax=43)
        self.assertEqual(trunc, SHTTruncation(85, 43, 43))
        self.assertEqual(trunc.lmax, trunc.mmax + trunc.lmmax - 1)
        for precompute in (th.legendre._precompute_legpoly, th.legendre._precompute_dlegpoly):
            dense = precompute(43, 85, 96, "legendre-gauss")
            actual = precompute(43, 85, 96, "legendre-gauss", trunc=trunc)
            for m in range(43):
                self.assertTrue(torch.equal(actual[..., m, m + 42, :], dense[..., m, m + 42, :]))
                self.assertTrue(actual[..., m, m + 42, :].ne(0).any())
                if m + 43 < 85:
                    self.assertTrue(torch.equal(actual[..., m, m + 43, :], torch.zeros_like(actual[..., m, m + 43, :])))

    def test_scalar_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 11, 4] = 1.0 + 0.25j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("scalar round trip", coeffs, actual, atol=1e-9, rtol=1e-9, verbose=verbose))

    @parameterized.expand([(9, 5, 5), (7, 5, 5)])
    def test_scalar_bandwidth_support(self, lmax, mmax, lmmax, verbose=False):
        nlat, nlon = 16, 32
        support = spectral_support(lmax, mmax, lmmax).transpose(0, 1)

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", lmmax=lmmax)
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", lmmax=lmmax)

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 4, 0] = 1.0
        coeffs[0, lmax - 1, 4] = 0.5 + 0.25j
        coeffs[0, 5, 0] = 3.0 - 2.0j
        coeffs[0, 6, 1] = -4.0 + 1.5j
        zeroed = coeffs.clone()
        zeroed[:, ~support] = 0.0

        with torch.no_grad():
            actual = isht(coeffs)
            reference = isht(zeroed)
        self.assertTrue(compare_tensors("invalid scalar inverse modes", reference, actual, atol=1e-12, rtol=1e-12, verbose=verbose))

        with torch.no_grad():
            round_trip = sht(isht(zeroed))
            random_output = sht(torch.randn(1, nlat, nlon, dtype=torch.float64))
        self.assertTrue(compare_tensors("retained scalar boundary modes", zeroed[..., support], round_trip[..., support], atol=1e-9, rtol=1e-9, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "scalar forward outside support",
                torch.zeros_like(random_output[..., ~support]),
                random_output[..., ~support],
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        )

    def test_vector_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, 11, 4] = 1.0 + 0.25j
        coeffs[0, 1, 10, 4] = -0.5 + 0.75j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("vector round trip", coeffs, actual, atol=1e-7, rtol=1e-7, verbose=verbose))

    @parameterized.expand([(9, 5, 5), (7, 5, 5)])
    def test_vector_bandwidth_support(self, lmax, mmax, lmmax, verbose=False):
        nlat, nlon = 16, 32
        support = spectral_support(lmax, mmax, lmmax).transpose(0, 1)

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", lmmax=lmmax)
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", lmmax=lmmax)

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, lmax - 1, 4] = 1.0 + 0.25j  # retained boundary mode
        coeffs[0, 1, lmax - 1, 4] = -0.5 + 0.75j
        coeffs[0, 0, 5, 0] = 3.0 - 2.0j  # l - m == lmmax
        coeffs[0, 1, 6, 1] = -4.0 + 1.5j  # l - m == lmmax
        zeroed = coeffs.clone()
        zeroed[..., ~support] = 0.0

        with torch.no_grad():
            actual = isht(coeffs)
            reference = isht(zeroed)
            round_trip = sht(reference)
            random_output = sht(torch.randn(1, 2, nlat, nlon, dtype=torch.float64))
        self.assertTrue(compare_tensors("invalid vector inverse modes", reference, actual, atol=1e-12, rtol=1e-12, verbose=verbose))
        self.assertTrue(compare_tensors("retained vector boundary modes", zeroed[..., support], round_trip[..., support], atol=1e-7, rtol=1e-7, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "vector forward outside support",
                torch.zeros_like(random_output[..., ~support]),
                random_output[..., ~support],
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        )


if __name__ == "__main__":
    unittest.main()
