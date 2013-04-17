try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import to_device

from neuron.lif import LIFNeuron

#from gemm_batched import gemm_batched_op as gemm_batched

############################
# Destined for connection.py
############################

class LowRankConnection(object):
    """
    dst_view.input_current += dot(u, dot(v, src_view.ouput))
    """
    def __init__(self, queue, src_view, dst_view, u, v):
        self.src_view = src_view 
        self.dst_view = dst_view
        self.u = u
        self.v = v
        #self.u = cl_array.to_device(queue, u)
        #self.v = cl_array.to_device(queue, v)

    @property
    def rank(self):
        return self.v.shape[0]


class BatchedLowRankConnection(object):
    def __init__(self, connections):
        conns = self.connections = list(connections)
        population = self.population = connections[0].src_view.population

        assert all(c.src_view.population == self.population for c in conns)
        assert all(c.dst_view.population == self.population for c in conns)

        def consolidate(aseq):
            u_all = np.empty(sum(a.size for a in aseq), dtype='float32')
            u_starts = []
            u_rows = []
            u_cols = []

            u_offset = 0
            for a in aseq:
                u_all[u_offset:u_offset + a.size] = a.flatten()
                u_starts.append(u_offset)
                u_rows.append(a.shape[0])
                u_cols.append(a.shape[1])
                u_offset += a.size
            return map(np.asarray, [u_all, u_starts, u_rows, u_cols])

        u_stuff = consolidate([c.u for c in conns])
        v_stuff = consolidate([c.v for c in conns])

        queue = self.population.queue

        cl_latent = cl_array.zeros(queue,
                                   (sum(c.rank for c in conns)),
                                   dtype='float32')
        cl_latent_starts = cl_array.to_device(queue,
                np.cumsum([0] + [c.rank for c in conns][:-1]))
        cl_latent_stuff = [cl_latent, cl_latent_starts]

        cl_u_stuff = [cl_array.to_device(queue, a) for a in u_stuff]
        cl_v_stuff = [cl_array.to_device(queue, a) for a in v_stuff]

        cl_x_stuff = [population.output,
                to_device(queue, np.asarray([c.src_view.start for c in conns])),
                ]
        # XXX verify that write areas do not overlap
        cl_y_stuff = [population.input_current,
                to_device(queue, np.asarray([c.dst_view.start for c in conns])),
                ]

        self._fn1 = self.compile_gemv_batched(queue.context, alpha=1.0, beta=0.0)
        self._fn2 = self.compile_gemv_batched(queue.context, alpha=1.0, beta=1.0)

        self._args1 = [cla.data for cla in (cl_u_stuff + cl_x_stuff + cl_latent_stuff)]
        self._args2 = [cla.data for cla in (cl_v_stuff + cl_latent_stuff + cl_y_stuff)]
        #self._fn1.set_args(*self._args1)
        #self._fn2.set_args(*self._args2)

    def compile_gemv_batched(self, context, alpha, beta):
        return cl.Program(context, """
            __kernel void fn(
                __global const float *X_data,
                __global const int *X_starts,
                __global const int *X_rows,
                __global const int *X_cols,

                __global const float *Y_data,
                __global const int *Y_starts,

                __global float *Z_data,
                __global const int *Z_starts
                         )
            {
                int gid = get_global_id(0);
                X_data += X_starts[gid];
                Y_data += Y_starts[gid];
                Z_data += Z_starts[gid];
                const int K = X_cols[gid];
                const float alpha = %(alpha)s;
                const float beta = %(beta)s;
                for (int mm = 0; mm < X_rows[gid]; ++mm)
                {
                    float ksum = 0.0;
                    for (int kk = 0; kk < K; ++kk)
                    {
                        ksum += X_data[kk] * Y_data[kk];
                    }
                    Z_data[mm] = beta * Z_data[mm] + alpha * ksum;
                    X_data += K;
                }
            }
            """ % locals()).build().fn

    def cl_update(self, queue):
        if 0:
            cl.enqueue_nd_range_kernel(queue, self._fn1, (len(self.connections),), None)
            cl.enqueue_nd_range_kernel(queue, self._fn2, (len(self.connections),), None)
        else:
            self._fn1(queue, (len(self.connections),), None, *self._args1)
            self._fn2(queue, (len(self.connections),), None, *self._args2)

    def add_to_updates(self, updates):
        v1_start = self.connections[0].v1.start
        v1_rowlen = len(self.connections[0].v1)

        v2_start = self.connections[0].v2.start
        v2_rowlen = len(self.connections[0].v2)

        v1len = len(self.connections) * v1_rowlen
        v2len = len(self.connections) * v2_rowlen

        voltage = self.population.voltage
        output = self.population.output

        # -- multiply neuron outputs by weights to increment voltage

        output3 = output[v1_start:v1_start + v1len].reshape(
                (len(self.connections), v1_rowlen, 1))

        # -- the transpose is a hack for speed on GPU
        decoded = gemm_batched(1.0, self.Ustack.transpose(0, 2, 1), output3)

        newvolt = updates.get(voltage)
        if newvolt is None:
            newvolt3 = None
        else:
            newvolt3 = newvolt[v2_start:v2_start + v2len].reshape(
                    (len(self.connections), v2_rowlen, 1))

        # -- the transpose is a hack for speed on GPU
        newvolt3 = gemm_batched(1.0, self.Vstack.transpose(0, 2, 1), decoded, 1.0, newvolt3)

        if newvolt is None:
            updates[voltage] = tensor.inc_subtensor(
                voltage[v2_start:v2_start + v2len],
                newvolt3.flatten())
        else:
            updates[voltage] = tensor.inc_subtensor(
                newvolt[v2_start:v2_start + v2len],
                newvolt3.flatten())
        return updates


def random_low_rank_connection(queue, v1, v2, rank, rng=None, dtype='float32'):
    if rng is None:
        rng = np.random
    u = rng.randn(rank, len(v1)).astype(dtype)
    v = rng.randn(len(v2), rank).astype(dtype)
    return LowRankConnection(queue, v1, v2, u, v)


def decoder_encoder(v1, v2, latent_v1, latent_v2, samples_v1, samples_v2, rng=None):
    # Given that neuron view v1 represents latent signal `latent_v1`
    # as
    pass


############################
# Destined for simulator.py
############################

class Simulator(object):
    def __init__(self, populations, connections):
        self.populations = populations
        self.connections = connections

        queue = self.populations[0].queue
        assert all(p.queue == queue for p in populations)

        # compress the set of connections as much as possible
        # TODO: make this a registry or smth
        self._conns = OrderedDict()
        for c in connections:
            self._conns.setdefault(type(c), []).append(c)

        if len(self._conns.get(LowRankConnection, [])) > 1:
            self._conns[LowRankConnection] = [BatchedLowRankConnection(
                self._conns[LowRankConnection])]
        self.queue = queue

    def step(self, n):
        queue = self.queue
        updates = [p.cl_update for p in self.populations]
        for ctype, clist in self._conns.items():
            updates.extend([c.cl_update for c in clist])
        for i in xrange(n):
            for update in updates:
                update(queue)
        queue.finish()

