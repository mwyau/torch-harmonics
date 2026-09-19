# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
from testutils import compare_tensors

import torch_harmonics as th
from torch_harmonics.truncation import _sht_truncation_mask


class TestSHTTruncation(unittest.TestCase):
    def test_default_is_triangular(self):
        self.assertEqual(th.truncate_sht(16, 16, grid="legendre-gauss"), (9, 9))
        self.assertEqual(th.truncate_sht(32, 16, grid="legendre-gauss"), (9, 9))

    def test_explicit_triangular(self):
        self.assertEqual(th.truncate_sht(16, 32, lmax=12, mmax=5, grid="legendre-gauss"), (5, 5))
        self.assertEqual(th.truncate_sht(32, 16, lmax=24, grid="legendre-gauss"), (9, 9))
        self.assertEqual(th.truncate_sht(32, 16, mmax=12, grid="legendre-gauss"), (12, 12))
        self.assertEqual(th.truncate_sht(16, 32, lmax=12, mmax=5, grid="legendre-gauss", truncation="triangular"), (5, 5))

        for cls in (th.RealSHT, th.InverseRealSHT, th.RealVectorSHT, th.InverseRealVectorSHT):
            default = cls(16, 32, lmax=12, mmax=5, grid="legendre-gauss")
            explicit = cls(16, 32, lmax=12, mmax=5, grid="legendre-gauss", truncation="triangular")
            self.assertEqual((default.lmax, default.mmax), (5, 5))
            self.assertEqual((explicit.lmax, explicit.mmax), (5, 5))

    def test_trapezoidal(self):
        cases = (
            ((32, 16), {}, (32, 9)),
            ((32, 16), {"lmax": 24}, (24, 9)),
            ((32, 16), {"mmax": 12}, (32, 12)),
            ((16, 32), {"lmax": 12, "mmax": 5}, (12, 5)),
            ((16, 32), {"lmax": 5, "mmax": 12}, (5, 5)),
        )
        for (nlat, nlon), limits, expected in cases:
            with self.subTest(nlat=nlat, nlon=nlon, limits=limits):
                self.assertEqual(
                    th.truncate_sht(nlat, nlon, grid="legendre-gauss", truncation="trapezoidal", **limits),
                    expected,
                )

    def test_rhomboidal_bounds_and_global_offsets(self):
        lmax, mmax = 9, 5  # standard R4 in torch-harmonics' non-inclusive limits
        self.assertEqual(
            th.truncate_sht(16, 32, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="rhomboidal"),
            (lmax, mmax),
        )
        self.assertEqual(
            th.truncate_sht(16, 32, lmax=5, mmax=9, grid="legendre-gauss", truncation="rhomboidal"),
            (5, 5),
        )

        support = _sht_truncation_mask(lmax, mmax, "rhomboidal")
        self.assertTrue(support[0, 4])  # (l=4, m=0), l-m=N
        self.assertTrue(support[4, 8])  # (l=8, m=4), l-m=N
        self.assertFalse(support[0, 5])  # (l=5, m=0), just outside the edge
        self.assertFalse(support[3, 8])  # (l=8, m=3), just outside the edge

        local = _sht_truncation_mask(
            lmax,
            mmax,
            "rhomboidal",
            mmin=3,
            lmin=4,
            local_mmax=5,
            local_lmax=9,
        )
        self.assertTrue(torch.equal(local, support[3:, 4:]))

    def test_invalid_truncation(self):
        with self.assertRaisesRegex(ValueError, "triangular.*trapezoidal"):
            th.truncate_sht(16, 32, truncation="invalid")

    def test_scalar_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 11, 4] = 1.0 + 0.25j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("scalar round trip", coeffs, actual, atol=1e-9, rtol=1e-9, verbose=verbose))

    def test_scalar_rhomboidal_support(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 9, 5
        support = _sht_truncation_mask(lmax, mmax, "rhomboidal").transpose(0, 1)

        sht = th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="rhomboidal")
        isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="rhomboidal")

        coeffs = torch.zeros(1, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 4, 0] = 1.0
        coeffs[0, 8, 4] = 0.5 + 0.25j
        coeffs[0, 5, 0] = 3.0 - 2.0j
        coeffs[0, 8, 3] = -4.0 + 1.5j
        zeroed = coeffs.clone()
        zeroed[:, ~support] = 0.0

        with torch.no_grad():
            actual = isht(coeffs)
            reference = isht(zeroed)
        self.assertTrue(compare_tensors("invalid scalar inverse modes", actual, reference, atol=1e-12, rtol=1e-12, verbose=verbose))

        with torch.no_grad():
            round_trip = sht(isht(zeroed))
            random_output = sht(torch.randn(1, nlat, nlon, dtype=torch.float64))
        self.assertTrue(compare_tensors("retained scalar boundary modes", round_trip[..., support], zeroed[..., support], atol=1e-9, rtol=1e-9, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "scalar forward outside rhomboid",
                random_output[..., ~support],
                torch.zeros_like(random_output[..., ~support]),
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        )

    def test_vector_trapezoidal_round_trip(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 12, 5

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="trapezoidal")

        self.assertEqual((sht.lmax, sht.mmax), (lmax, mmax))
        self.assertEqual((isht.lmax, isht.mmax), (lmax, mmax))

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, 11, 4] = 1.0 + 0.25j
        coeffs[0, 1, 10, 4] = -0.5 + 0.75j

        with torch.no_grad():
            actual = sht(isht(coeffs))

        self.assertTrue(compare_tensors("vector round trip", coeffs, actual, atol=1e-7, rtol=1e-7, verbose=verbose))

    def test_vector_rhomboidal_support_and_halo(self, verbose=False):
        nlat, nlon = 16, 32
        lmax, mmax = 9, 5
        support = _sht_truncation_mask(lmax, mmax, "rhomboidal").transpose(0, 1)

        sht = th.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="rhomboidal")
        isht = th.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss", truncation="rhomboidal")

        coeffs = torch.zeros(1, 2, lmax, mmax, dtype=torch.complex128)
        coeffs[0, 0, 8, 4] = 1.0 + 0.25j  # retained high-degree/high-order boundary
        coeffs[0, 1, 8, 4] = -0.5 + 0.75j
        coeffs[0, 0, 5, 0] = 3.0 - 2.0j  # invalid because l-m=N+1
        coeffs[0, 1, 8, 3] = -4.0 + 1.5j  # invalid because l-m=N+1
        zeroed = coeffs.clone()
        zeroed[..., ~support] = 0.0

        with torch.no_grad():
            actual = isht(coeffs)
            reference = isht(zeroed)
            round_trip = sht(reference)
            random_output = sht(torch.randn(1, 2, nlat, nlon, dtype=torch.float64))
        self.assertTrue(compare_tensors("invalid vector inverse modes", actual, reference, atol=1e-12, rtol=1e-12, verbose=verbose))
        self.assertTrue(compare_tensors("retained vector boundary modes", round_trip[..., support], zeroed[..., support], atol=1e-7, rtol=1e-7, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "vector forward outside rhomboid",
                random_output[..., ~support],
                torch.zeros_like(random_output[..., ~support]),
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        )


if __name__ == "__main__":
    unittest.main()
