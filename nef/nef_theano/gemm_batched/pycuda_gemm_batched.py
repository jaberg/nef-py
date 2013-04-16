import os
import numpy as np
import theano
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.misc import pycuda_init
if not pycuda_init.pycuda_available:
    raise Exception("No pycuda available. You can't load pycuda_example.py")

from pycuda.compiler import SourceModule

def ints(*seq):
    return map(np.intc, seq)


class Base(object):
    def read_support_code(self):
        if __file__.endswith('pyc'):
            base = __file__[:-1]
        else:
            base = __file__
        cuname = os.path.join(os.path.dirname(base), self.__class__.__name__ + '.cu')
        return open(cuname).read()

    def call_checks(self, a, X, Y, b, Z):
        B, M, K = X.shape
        _B, _K, N = Y.shape
        if (B,)  != a.shape:
            raise ValueError('a had wrong shape')
        if (B, K, N) != Y.shape:
            raise ValueError('Y had wrong shape')
        if (B,) != b.shape:
            raise ValueError('b had wrong shape')
        if (B, M, N) != Z.shape:
            raise ValueError('Z had wrong shape')
        return ints(B, M, N, K)


class RefAlgo(Base):
    def __init__(self):
        self.mod = SourceModule(self.read_support_code())
        self.kernel = self.mod.get_function('refk')

    def __call__(self, a, X, Y, b, Z):
        B, M, N, K = self.call_checks(a, X, Y, b, Z)
        sa0, = ints(a._strides)
        sb0, = ints(b._strides)
        sx0, sx1, sx2 = ints(*X._strides)
        sy0, sy1, sy2 = ints(*Y._strides)
        sz0, sz1, sz2 = ints(*Z._strides)

        args = (B, M, N, K,
                    a, sa0,
                    X, sx0, sx1, sx2,
                    Y, sy0, sy1, sy2, 
                    b, sb0,
                    Z, sz0, sz1, sz2,
               )
        self.kernel(grid=(int(B), 1), block=(16, 16, 1), *args)

ref_algo = RefAlgo()


def autoselect_gemm_batched(a, X, Y, b, Z):
    ref_algo(a, X, Y, b, Z)

from gemm_batched import GemmBatched
from gpu_gemm_batched import register_opt, local_optimizer
from gpu_gemm_batched import GpuGemmBatched

class PyCudaGemmBatched(GpuOp, GemmBatched):

    def make_node(self, *inputs):
        _inputs = map(as_cuda_ndarray_variable, inputs)
        otype = _inputs[-1].type
        return theano.Apply(self, _inputs, [otype()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        sa, sX, sY, sb, sZ = node_input_storage
        sZout, = node_output_storage
        cZout, = node_output_compute
        def thunk():
            # TODO: skip copy if destructive 
            # and neither sZout nor sZ in no_recyling
            # TODO: re-use Zout storage if sZout not in no_recyling and Zout
            # has right dims etc.
            Z = sZ[0].copy()
            autoselect_gemm_batched(sa[0], sX[0], sY[0], sb[0], Z)
            sZout[0] = Z
            cZout[0] = True
        thunk.node_input_storage = node_input_storage
        thunk.node_output_storage = node_output_storage
        thunk.node_input_compute = node_input_compute
        thunk.node_output_compute = node_output_compute
        thunk.lazy = False
        return thunk

@register_opt()
@local_optimizer()
def use_pycuda_gemm_batched(node):
    if (isinstance(node.op, GpuGemmBatched)
            and node.outputs[0].dtype == 'float32'):
        op = PyCudaGemmBatched(node.op.destructive)
        return [op(*node.inputs)]
