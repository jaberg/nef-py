try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import to_device

from neuron.lif import LIFNeuron

from gemm_batched.ocl_gemm_batched import choose_gemv_batched_plan

#from gemm_batched import gemm_batched_op as gemm_batched

############################
# Destined for connection.py
############################

class LowRankConnection(object):
    """
    dst_view.input_current += dot(enc, dot(dec, src_view.ouput))
    """
    def __init__(self, queue, src_view, dst_view, dec, enc):
        self.src_view = src_view 
        self.dst_view = dst_view
        self.dec = dec
        self.enc = enc

    @property
    def rank(self):
        return self.enc.shape[1]


class BatchedLowRankConnection(object):
    """
    for i, a, b, c, d, e, f in ranges(...):
        dst[a:b] <- dot(enc[i], dot(dec[i], src[c:d]))

        # As a side-effect, stores
        decoded[e:f] <- dot(dec[i], src[c:d])

    This is implemented as two "batched gemv" calls, so it's essential that the
    various dst[a:b] ranges do not overlap.

    """
    def __init__(self, connections):
        conns = self.connections = list(connections)
        population = self.population = connections[0].src_view.population
        queue = self.population.queue

        assert all(c.src_view.population == self.population for c in conns)
        assert all(c.dst_view.population == self.population for c in conns)

        conns.sort(lambda a, b: cmp(a.dst_view.start, b.dst_view.start))
        # -- check that we are working with non-overlapping outputs
        for ci, cj in zip(conns[:], conns[1:]):
            if ci.dst_view.stop > cj.dst_view.start:
                raise NotImplementedError('overlapping outputs')

        try:
            dec_stack = np.asarray([c.dec for c in conns])
            enc_stack = np.asarray([c.enc for c in conns])
        except:
            # -- actually, not implemented would be OK because
            #    the kernel can technically deal with different dims
            raise

        self.dec_stack = to_device(queue, dec_stack)
        self.enc_stack = to_device(queue, enc_stack)

        self.decoded = cl_array.zeros(queue,
                                      sum(c.rank for c in conns),
                                      dtype='float32')

        self.X_dec_offsets = to_device(queue,
            np.array([c.src_view.start for c in conns], dtype='intc'))
        self.Y_enc_offsets = to_device(queue,
            np.array([c.dst_view.start for c in conns], dtype='intc'))
        self.decoded_offsets = to_device(queue,
            np.arange(0, self.decoded.shape[0], conns[0].rank).astype('intc'))


        if not all(c.src_view.step == 1 for c in conns):
            raise NotImplementedError()
        if not all(c.dst_view.step == 1 for c in conns):
            raise NotImplementedError()


        decshp = self.dec_stack.shape
        encshp = self.enc_stack.shape

        assert conns[0].rank == decshp[1] == encshp[2]

        self.dec_plan = choose_gemv_batched_plan(
            BMN=dec_stack.shape,
            alpha=1.0,
            Aparams=(self.dec_stack, 0, decshp[1] * decshp[2], decshp[2], 1),
            Xparams=(population.output, self.X_dec_offsets, 1),
            beta=0.0,
            Yparams=(self.decoded, self.decoded_offsets, 1),
            queues=[queue])

        self.enc_plan = choose_gemv_batched_plan(
            BMN=enc_stack.shape,
            alpha=1.0,
            Aparams=(self.enc_stack, 0, encshp[1] * encshp[2], encshp[2], 1),
            Xparams=(self.decoded, self.decoded_offsets, 1),
            beta=1.0,
            Yparams=(population.input_current, self.Y_enc_offsets, 1),
            queues=[queue])


    def cl_update(self, queue):
        self.dec_plan()
        self.enc_plan()


def random_low_rank_connection(queue, v1, v2, rank, rng=None, dtype='float32'):
    if rng is None:
        rng = np.random
    dec = rng.randn(rank, len(v1)).astype(dtype)
    enc = rng.randn(len(v2), rank).astype(dtype)
    return LowRankConnection(queue, v1, v2, dec, enc)


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

