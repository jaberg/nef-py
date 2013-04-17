import unittest

import numpy as np

import theano
from theano import tensor

from nef.nef_theano.gemm_batched import gemm_batched_op

def define_smokes(seq):
    for s in seq:
        (B, M, N, K), dtype = s
        def test_smoke():
            a = np.random.randn(B).astype(dtype)
            X = np.random.randn(B, M, K).astype(dtype)
            Y = np.random.randn(B, K, N).astype(dtype)
            b = np.random.randn(B).astype(dtype)
            Z = np.random.randn(B, M, N).astype(dtype)

            sa, sb = tensor.vectors('ab')
            sX, sY, sZ = tensor.tensor3s('XYZ')

            sR = gemm_batched_op(sa, sX, sY, sb, sZ)
            f = theano.function([sa, sX, sY, sb, sZ], sR,
                               mode='DEBUG_MODE')

            R = f(a, X, Y, b, Z)
        test_smoke.__name__ += ' '.join(map(str, [B, M, N, K, dtype]))
        globals()[test_smoke.__name__] = test_smoke

define_smokes([
    ((1, 1, 1, 1), 'float32'),
    ((2, 1, 1, 1), 'float32'),
    ((1, 2, 1, 1), 'float32'),
    ((1, 1, 2, 1), 'float32'),
    ((1, 1, 1, 2), 'float32'),
    ((3, 3, 1, 1), 'float32'),
    ((2, 3, 4, 5), 'float32'),
    ((2, 9, 8, 7), 'float32'), # -- bigger than 256 (see cgemm_batched_py.c:195)
])

