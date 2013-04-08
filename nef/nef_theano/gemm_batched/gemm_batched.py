import numpy as np
from theano import Op, Apply
from theano import tensor

class GemmBatched(Op):
    """
    Op provides Python equivalent of CUDA's gemmBatched function.

    It is more restricted than CUDA's gemmBatched for now in requiring the
    arguments to be the slices of a 3D tensor, rather than simply a list of
    matrices.
    """

    def __init__(self, destructive):
        self._attrs = (destructive,)
        if destructive:
            self.destroy_map = {0: [4]}
        else:
            self.destroy_map = {}

    def __eq__(self, other):
        return type(self) == type(other) and self._attrs == other._attrs

    def __hash__(self):
        return hash((type(self), self._attrs))

    def make_node(self, a, X, Y, b, Z):
        a, X, Y, b, Z  = map(tensor.as_tensor_variable, [a, X, Y, b, Z])
        assert a.ndim == 1
        assert X.ndim == 3
        assert Y.ndim == 3
        assert b.ndim == 1
        assert Z.ndim == 3
        # TODO: check dims and dtypes
        return Apply(self, [a, X, Y, b, Z], [Z.type()])

    def perform(self, node, inputs, outstor):
        a, X, Y, b, Z = inputs
        B, M, N = Z.shape
        B_, M_, K = X.shape
        #print 'PERFORM', B, M, N, K

        assert (B,) == a.shape
        assert (B, M, K) == X.shape
        assert (B, K, N) == Y.shape, ((B, K, N), Y.shape)
        assert (B,) == b.shape

        if not self.destroy_map:
            Z = Z.copy()
        outstor[0][0] = Z

        for Zi, ai, Xi, Yi, bi in zip(Z, a, X, Y, b):
            if bi == 1.0:
                Zi += ai * np.dot(Xi, Yi)
            elif bi == 0.0:
                Zi[:] = ai * np.dot(Xi, Yi)
            else:
                Zi *= bi
                Zi += ai * np.dot(Xi, Yi)

    def __call__(self, a, X, Y, b=0.0, Z=None):
        a, X, Y, b = map(tensor.as_tensor_variable, [a, X, Y, b])
        if a.ndim == 0:
            a = a + tensor.zeros_like(X[:, 0, 0])

        if b.ndim == 0:
            b = b + tensor.zeros_like(X[:, 0, 0])

        if Z is None:
            Z = tensor.zeros([X.shape[0], X.shape[1], Y.shape[2]],
                dtype=X.dtype)

        op = GemmBatched(destructive=False)
        return Op.__call__(self, a, X, Y, b, Z)


gemm_batched_op = GemmBatched(False)

