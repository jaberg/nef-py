
from theano.tensor.blas_c import BaseBLAS
from theano.tensor.blas_c import optdb, blas_optdb
from theano.tensor.blas_c import local_optimizer
from gemm_batched import GemmBatched
from gemm_batched import gemm_batched_op

class CGemmBatched(BaseBLAS, GemmBatched):
    def c_compile_args(self):
        rval = BaseBLAS.c_compile_args(self)
        rval.append('-ftree-vectorize')
        rval.append('-funsafe-math-optimizations')
        # -- this is doing only harm in test_runtime_api2, using openblas
        #    on ctn00 circa April 2013.
        #rval.append('-fopenmp')
        return rval

    def __str__(self):
        d = 'destructive' if self.destroy_map else 'non-destructive'
        return 'CGemmBatched{%s}' % d

    def c_support_code_apply(self, node, name):
        rval1 = BaseBLAS.c_support_code(self)
        path = __file__
        if path.endswith('pyc'):
            path = path[:-1]
        rval2 = open(path + '.support.c').read() % locals()
        return rval1 + '\n' + rval2

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
        return (10, blas_header_version())


@local_optimizer()
def use_c_gemm_batched(node):
    if (node.op == gemm_batched_op and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGemmBatched(False)(*node.inputs)]


@local_optimizer()
def make_c_gemmbatched_destructive(node):
    if node.op == CGemmBatched(False):
        return [CGemmBatched(True)(*node.inputs)]


blas_optdb['use_c_blas'].local_optimizers.append(use_c_gemm_batched)
optdb['c_blas_destructive'].local_optimizers.append(make_c_gemmbatched_destructive)
