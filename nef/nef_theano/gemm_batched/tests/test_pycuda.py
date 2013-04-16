import unittest

import numpy as np

import theano
from theano import tensor

from theano.sandbox.cuda import CudaNdarray
from nef.nef_theano.gemm_batched import pycuda_gemm_batched

def np_gemm_batched(a, X, Y, b, Z):
    for ai, xi, yi, bi, zi in zip(a, X, Y, b, Z):
        zi *= bi
        zi += ai * np.dot(xi, yi)


def define_gemm_batched_impl_checks(cls, seq):
    for s in seq:
        (B, M, N, K), dtype = s
        def test_gemm_batched_impl():
            a = np.random.randn(B).astype(dtype)
            X = np.random.randn(B, M, K).astype(dtype)
            Y = np.random.randn(B, K, N).astype(dtype)
            b = np.random.randn(B).astype(dtype)
            Z = np.random.randn(B, M, N).astype(dtype)

            ga, gX, gY, gb, gZ = map(CudaNdarray, [a, X, Y, b, Z])
            assert np.allclose(Z, gZ)

            impl = cls()
            impl(ga, gX, gY, gb, gZ)
            np_gemm_batched(a, X, Y, b, Z)

            assert np.allclose(Z, gZ, atol=1e-4, rtol=1e-4)

        test_gemm_batched_impl.__name__ += '_'.join(map(str,
                                            [cls.__name__, B, M, N, K, dtype]))
        globals()[test_gemm_batched_impl.__name__] = test_gemm_batched_impl

define_gemm_batched_impl_checks(pycuda_gemm_batched.RefAlgo, [
    ((1, 1, 1, 1), 'float32'),
    ((2, 1, 1, 1), 'float32'),
    ((1, 2, 1, 1), 'float32'),
    ((1, 1, 2, 1), 'float32'),
    ((1, 1, 1, 2), 'float32'),
    ((3, 3, 1, 1), 'float32'),
    ((2, 3, 4, 5), 'float32'),
    ((2, 9, 8, 7), 'float32'), # TODO put some bigger ones
])


def define_op_checks(seq):
    for s in seq:
        (B, M, N, K), dtype = s
        def test_pycuda_gemm_batched_nondestructive():
            a = np.random.randn(B).astype(dtype)
            X = np.random.randn(B, M, K).astype(dtype)
            Y = np.random.randn(B, K, N).astype(dtype)
            b = np.random.randn(B).astype(dtype)
            Z = np.random.randn(B, M, N).astype(dtype)

            sa, sb = tensor.vectors('ab')
            sX, sY, sZ = tensor.tensor3s('XYZ')

            sR = pycuda_gemm_batched.PyCudaGemmBatched()(sa, sX, sY, sb, sZ)
            f = theano.function([sa, sX, sY, sb, sZ], sR)

            R = f(a, X, Y, b, Z)

            np_gemm_batched(a, X, Y, b, Z)
            assert np.allclose(R, Z, atol=1e-4, rtol=1e-4)

        test_pycuda_gemm_batched_nondestructive.__name__ += ' '.join(map(str, [B, M, N, K, dtype]))
        globals()[test_pycuda_gemm_batched_nondestructive.__name__] = test_pycuda_gemm_batched_nondestructive

define_op_checks([
    ((1, 1, 1, 1), 'float32'),
    ((2, 1, 1, 1), 'float32'),
    ((1, 2, 1, 1), 'float32'),
    ((1, 1, 2, 1), 'float32'),
    ((1, 1, 1, 2), 'float32'),
    ((3, 3, 1, 1), 'float32'),
    ((2, 3, 4, 5), 'float32'),
    ((2, 9, 8, 7), 'float32'), # -- bigger than 256 (see cgemm_batched_py.c:195)
])

