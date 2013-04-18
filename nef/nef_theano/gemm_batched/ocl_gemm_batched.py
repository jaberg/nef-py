import numpy as np
import pyopencl as cl


def gemm_batched_check_shapes(a, X, Y, b, Z):
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
    return B, M, N, K


class GemmBatchedPlan(object):
    def __init__(self, a, X, Y, b, Z):
        pass


def gemv_batched_ref(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets
                         )
        {
            const int bb = get_global_id(0);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];

            for (int mm = 0; mm < %(M)s; ++mm)
            {
                float ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(AsN)s  + mm * %(AsM)s] * X_data[nn * %(XsN)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(YsM)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(YsM)s * mm] = %(beta)s * Y_data[%(YsM)s * mm] + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn


class GemvBatchedPlan(object):

    def __init__(self, dct):
        self.__dict__.update(dct)

    def __call__(self):
        self._fn(*self._fn_args)
        


def choose_gemv_batched_plan(
    BMN, alpha, Aparams, Xparams, beta, Yparams, queues,
    ):
    B, M, N = BMN
    A_buf, Aoffset, AsB, AsM, AsN = Aparams
    X_buf, X_offsets, XsN = Xparams
    Y_buf, Y_offsets, YsM = Yparams
    queue, = queues
    if np.float32 != A_buf.dtype:
        raise NotImplementedError()
    if np.float32 != X_buf.dtype:
        raise NotImplementedError()
    if np.float32 != Y_buf.dtype:
        raise NotImplementedError()

    _fn = gemv_batched_ref(queue.context,
                                B, M, N,
                                alpha,
                                Aoffset, AsB, AsM, AsN,
                                XsN,
                                beta,
                                YsM)
    global_shape = (B,)
    local_shape = None
    _fn_args = (queue, global_shape, local_shape, A_buf.data,
                X_buf.data, X_offsets.data,
                Y_buf.data, Y_offsets.data)

    return GemvBatchedPlan(locals())
