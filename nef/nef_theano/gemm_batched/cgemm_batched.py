
from theano.tensor.blas_c import BaseBLAS
from theano.tensor.blas_c import opt_c_blas_destructive
from theano.tensor.blas_c import use_c_blas
from gemm_batched import GemmBatched

class CGemmBatched(BaseBLAS, GemmBatched):
    def c_support_code_apply(self, node, name):
        rval1 = BaseBLAS.c_support_code_apply(self, node, name)
        rval2 = open(__file__ + '_support.c').read() % locals()
        return rval1 + '\n' + rval2

    def c_code(self, node, name, inp, out, sub):
        src = open(__file__ + '.c').read()
        alpha, X, Y, beta, Z = inp
        zz, = out
        fail = sub['fail']
        return src % locals()

    def c_code_cache_version(self):
        return ()
        return (10, blas_header_version())


@local_optimizer()
def use_c_gemm_batched(node):
    if (node.op == GemmBatched() and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGemmBatched(False)(*node.inputs)]


@local_optimizer()
def make_c_gemmbatched_destructive(node):
    if node.op == CGemmBatched(False):
        return [CGemmBatched(True)(*node.inputs)]


use_c_blas.local_optimizer.append(use_c_gemm_batched)
opt_c_blas_destructive.local_optimizer.append(make_c_gemmbatched_destructive)
