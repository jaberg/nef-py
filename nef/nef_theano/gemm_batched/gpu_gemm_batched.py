
from theano import Apply, Op
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda import host_from_gpu
from theano.sandbox.cuda import gpu_from_host
#from theano.sandbox.cuda import GpuElemwise
from theano.sandbox.cuda.opt import register_opt

from gemm_batched import GemmBatched
from cgemm_batched import CGemmBatched
from theano.tensor.blas_c import local_optimizer

class GpuGemmBatched(GpuOp, GemmBatched):

    #def c_headers(self):
        #return GpuOp.c_headers(self) + ['cublas_v2.h']

    def make_node(self, *inputs):
        return Apply(self, inputs, [inputs[-1].type()])

    def __call__(self, *inputs):
        return Op.__call__(self, *inputs)

    def c_support_code_apply(self, node, name):
        rval0 =  """
#include <cublas_v2.h>
        """
        rval1 = ""
        path = __file__
        if path.endswith('pyc'):
            path = path[:-1]
        rval2 = open(path + '.support.c').read() % locals()
        return '\n'.join([rval0, rval1, rval2])


    def c_code(self, node, name, inp, out, sub):
        path = __file__
        if path.endswith('pyc'):
            path = path[:-1]
        src = open(path + '.c').read()
        alpha, X, Y, beta, Z = inp
        zz, = out
        destructive = int(bool(self.destroy_map))
        fail = sub['fail']
        return src % locals()

    def c_code_cache_version(self):
        return ()


@register_opt()
@local_optimizer()
def use_gpu_gemm_batched(node):
    if (isinstance(node.op, CGemmBatched)
            and node.outputs[0].dtype == 'float32'):
        op = GpuGemmBatched(node.op.destructive)
        rval = op(*map(gpu_from_host, node.inputs))
        return [host_from_gpu(rval)]

#@register_opt()
@local_optimizer()
def use_destructive_gpu_gemm_batched(node):
    if (isinstance(node.op, GpuGemmBatched)
            and not node.op.destructive):
        op = GpuGemmBatched(True)
        return [op(*node.inputs)]

#@register_opt()
@local_optimizer()
def at_least_copy_on_device(node):
    if node.op == gpu_from_host:
        hx = node.inputs[0]
        if hx.owner and hx.owner.op == host_from_gpu:
            op = GpuElemwise(theano.scalar.ops.second)
            return [op(hx.owner.inputs[0])]

