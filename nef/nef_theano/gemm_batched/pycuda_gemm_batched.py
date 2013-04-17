import os
import numpy as np
import theano
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.misc import pycuda_init
if not pycuda_init.pycuda_available:
    raise Exception("No pycuda available. You can't load pycuda_example.py")
# -- currently theano just uses the autoinit mechanism
import pycuda.autoinit

from pycuda.compiler import SourceModule
from pycuda._pvt_struct import pack


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


def args_std_checks(a, X, Y, b, Z):
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
    B, M, N, K = ints(B, M, N, K)
    sa0, = ints(*a._strides)
    sb0, = ints(*b._strides)
    sx0, sx1, sx2 = ints(*X._strides)
    sy0, sy1, sy2 = ints(*Y._strides)
    sz0, sz1, sz2 = ints(*Z._strides)

    return map(int, (B, M, N, K)), (B, M, N, K,
            int(np.intp(a.gpudata)), sa0,
            int(np.intp(X.gpudata)), sx0, sx1, sx2,
            int(np.intp(Y.gpudata)), sy0, sy1, sy2, 
            int(np.intp(b.gpudata)), sb0,
            int(np.intp(Z.gpudata)), sz0, sz1, sz2,)


class AlgoRef(Base):
    def __init__(self):
        self.mod = SourceModule(self.read_support_code())
        self.kernel = self.mod.get_function('refk')

    def __call__(self, args):
        B = int(args[0])

algo_ref = AlgoRef()
device = pycuda.autoinit.device

class AlgoN1(Base):
    def __init__(self):
        self.mod = SourceModule(self.read_support_code())
        self.kern = self.mod.get_function('kern')
        self.kern_smallB = self.mod.get_function('kern_smallB')

algo_n1 = AlgoN1()


arg_format = 'iiiiPiPiiiPiiiPiPiii'
class AutoselectGemmBatched(object):
    def __init__(self, a, X, Y, b, Z):
        (B, M, N, K), args = args_std_checks(a, X, Y, b, Z)
        self.arg_buf = pack(arg_format, *args)
        if N == 1:
            self.shared = int(K) * 4
            self.block = (max(32, min(int(M), device.max_block_dim_x)), 1, 1)
            self.grid = (min(int(B), device.max_grid_dim_x), 1)

            if B <= device.max_grid_dim_x:
                self._fn = algo_n1.kern_smallB._launch_kernel
            else:
                self._fn = algo_n1.kern._launch_kernel
        else:
            raise NotImplementedError()

        self._fn_args = (self.grid, self.block, self.arg_buf, self.shared, None)

    def __call__(self):
        self._fn(*self._fn_args)

def gemm_batched(a, X, Y, b, Z):
    return AutoselectGemmBatched(a, X, Y, b, Z)()

from gemm_batched import GemmBatched
from gpu_gemm_batched import register_opt, local_optimizer
from gpu_gemm_batched import GpuGemmBatched

# TODO: implement this in C
# TODO: create a C calling convention for repeated Thunk calls on the same
#       physical memory
def io_sig(istormap, ostormap):
    isig = [(np.intp(a[0].gpudata), a[0].shape, a[0]._strides)
            for a in istormap]
    osig = [(np.intp(a[0].gpudata), a[0].shape, a[0]._strides)
            for a in ostormap]
    return isig, osig


class Thunk(object):
    def __init__(self, destructive, node, storage_map, compute_map):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        self.sa, self.sX, self.sY, self.sb, self.sZ = node_input_storage
        self.sZout, = node_output_storage
        self.cZout, = node_output_compute
        self.node_input_storage = node_input_storage
        self.node_output_storage = node_output_storage
        self.node_input_compute = node_input_compute
        self.node_output_compute = node_output_compute
        self.lazy = False
        self.destructive = destructive
        self.autosel = None
        self.autosel_sig = None
        assert self.destructive

    def __call__(self):
        Z = self.sZ[0]
        if self.destructive:
            self.sZout[0] = Z
            #cur_sig = io_sig(
                #self.node_input_storage,
                #self.node_output_storage)
            #if self.autosel is None or self.autosel_sig != cur_sig:
                #print 'Recalculting kernel and args', self
                #print self.autosel_sig
                #print cur_sig
            gemm_batched(self.sa[0], self.sX[0], self.sY[0], self.sb[0], Z)
            #self.autosel_sig = cur_sig
            #self.autosel()
        else:
            # TODO: try to re-use current output
            Z = Z.copy()
            AutoselectGemmBatched(self.sa[0], self.sX[0], self.sY[0], self.sb[0], Z)()
            self.sZout[0] = Z
        self.cZout[0] = True


class PyCudaGemmBatched(GpuOp, GemmBatched):

    def make_node(self, *inputs):
        _inputs = map(as_cuda_ndarray_variable, inputs)
        otype = _inputs[-1].type
        return theano.Apply(self, _inputs, [otype()])

    def __call__(self, *inputs):
        return theano.Op.__call__(self, *inputs)

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        destructive = self.destructive and (node.outputs[0] not in no_recycling)
        return Thunk(destructive, node, storage_map, compute_map)

@register_opt()
@local_optimizer()
def use_pycuda_gemm_batched(node):
    if (isinstance(node.op, GpuGemmBatched)
            and node.outputs[0].dtype == 'float32'):
        op = PyCudaGemmBatched(node.op.destructive)
        return [op(*node.inputs)]

@register_opt()
@local_optimizer()
def use_destructive_pycuda_gemm_batched(node):
    if (node.op == PyCudaGemmBatched(False)):
        return [PyCudaGemmBatched(True)(*node.inputs)]

