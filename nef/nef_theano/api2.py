try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import math

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import to_device

from neuron.lif import OCL_LIFNeuron

from gemm_batched.ocl_gemm_batched import choose_gemv_batched_plan

#from gemm_batched import gemm_batched_op as gemm_batched

############################
# Destined for connection.py
############################


class FullConnection(object):
    def __init__(self, queue, src_view, dst_view, W):
        self.src_view = src_view 
        self.dst_view = dst_view

        self.W = cl.array.to_device(queue, W.astype('float32'))
        try:
            cl_x = self.src_view.output
        except AttributeError:
            cl_x = self.src_view.population.output
        cl_y = self.dst_view.population.input_current

        self._x_offset = cl.array.to_device(queue,
            np.asarray([self.src_view.start], dtype='intc'))
        self._y_offset = cl.array.to_device(queue,
            np.asarray([self.dst_view.start], dtype='intc'))

        self.dec_plan = choose_gemv_batched_plan(
            BMN=(1,) + W.shape,
            alpha=1.0,
            Aparams=(self.W, 0, W.shape[0] * W.shape[1], W.shape[1], 1),
            Xparams=(cl_x, self._x_offset, 1),
            beta=0.0,
            Yparams=(cl_y, self._y_offset, 1),
            queues=[queue])


    def cl_update(self, queue):
        # TODO: a proper matrix-vector multiply
        self.dec_plan()

    @property
    def src_population(self):
        return self.src_view.population

    @property
    def dst_population(self):
        return self.dst_view.population


class LowRankConnection(object):
    """
    dst_view.input_current += dot(enc, dot(dec, src_view.ouput))
    """
    def __init__(self, queue, src_view, dst_view, dec, enc):
        self.src_view = src_view 
        self.dst_view = dst_view
        self.dec = dec
        self.enc = enc
        self.hack = BatchedLowRankConnection([self])

    @property
    def rank(self):
        return self.enc.shape[1]

    @property
    def src_population(self):
        return self.src_view.population

    @property
    def dst_population(self):
        return self.dst_view.population

    def cl_update(self, *args):
        return self.hack.cl_update(*args)


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


def random_connection(queue, src, dst, rng=np.random):
    W = rng.randn(len(dst), len(src))
    return FullConnection(queue, src, dst, W)


def decoder_encoder_connection(queue, src, dst, func,
                               rank=1, rng=np.random, dtype='float32'):
    dec = rng.randn(rank, len(src)).astype(dtype)
    enc = rng.randn(len(dst), rank).astype(dtype)
    rval = LowRankConnection(queue, src, dst, dec, enc)
    rval.func = func
    return rval




############################
# Destined for simulator.py
############################

class Simulator(object):
    def __init__(self, populations, connections):
        self.populations = populations
        self.connections = connections

        # compress the set of connections as much as possible
        # TODO: make this a registry or smth
        self._conns = OrderedDict()
        for c in connections:
            self._conns.setdefault(type(c), []).append(c)

        if len(self._conns.get(LowRankConnection, [])) > 1:
            c_batched = []
            c_rest = []
            for c in self._conns[LowRankConnection]:
                try:
                    batched = BatchedLowRankConnection(c_batched + [c])
                    c_batched += [c]
                except NotImplementedError:
                    c_rest += [c]
            #TODO: try to make a new batched op out of c_rest, etc.

            if len(c_batched) > 1:
                self._conns[LowRankConnection] = [batched] + c_rest

    def step(self, queue, n, dt):
        # XXX use dt
        updates = [p.cl_update for p in self.populations]
        for ctype, clist in self._conns.items():
            updates.extend([c.cl_update for c in clist])
        for i in xrange(n):
            for update in updates:
                update(queue)
        queue.finish()


class FuncInput(object):
    def __init__(self, queue, function, zero_after=None, name=None):
        self.t = 0
        self.function = function
        val = function(0)
        self.decoded = cl.array.to_device(queue,
                                          np.asarray([val], dtype='float32'))
        self.population = None

    @property
    def output(self):
        return self.decoded

    def __len__(self):
        return 1

    @property
    def start(self):
        return 0

    def cl_update(self, queue, dt):
        self.t += dt # TODO: should the dt increment go after?
        if self.zero_after is not None and self.t > self.zero_after:
            self.decoded.fill(queue, 0.0)
        else:
            self.decoded.fill(queue, self.function(self.t))


class ConnectionList(object):
    def __init__(self, connections=None):
        if connections is None:
            self.connections = []
        else:
            self.connections = connections

    def add(self, connection):
        self.connections.append(connection) 

    def simulator(self):
        populations = [None]
        for c in self.connections:
            if c.src_population not in populations:
                populations.append(c.src_population)
            if c.dst_population not in populations:
                populations.append(c.dst_population)
            if populations[-1] is None:
                populations.pop()
        # remove the None
        populations = populations[1:]
        return Simulator(populations, self.connections)

    def solve_decoder_encoders(self, queue):
        # -- iterate over connections 
        # -- some sort of toposort should be used for DAGs right?
        pass


class Network(object):
    def __init__(self, name, queue, dt=0.001):
        self.name = name
        self.queue = queue
        self.objects = OrderedDict()
        self.lif_pop = OCL_LIFNeuron(queue, 0)
        self.connections = OrderedDict()
        self.dt = dt
        self.simtime = 0.0

    def make_input(self, name, value):
        if callable(value):
            self.objects[name] = FuncInput(self.queue, function=math.sin)
        else:
            raise NotImplementedError()

    def make(self, name, neurons, dimensions):
        if dimensions == 1:
            start = len(self.lif_pop)
            self.lif_pop.extend(self.queue, neurons)
            self.objects[name] = self.lif_pop[start:start + neurons]
        else:
            raise NotImplementedError()

    def connect(self, src_name, dst_name, func=lambda x: x):
        src = self.objects[src_name]
        dst = self.objects[dst_name]
        if isinstance(src, FuncInput):
            conn = random_connection(self.queue, src, dst)
        else:
            conn = decoder_encoder_connection(self.queue, src, dst, func)
        # TODO: support multiple connections w same src and dst?
        self.connections[(src_name, dst_name)] = conn

        # XXX solve decoders right here using self.dt

    def run(self, requested_simtime, rebuild_simulator=True):
        n_steps = int(requested_simtime / self.dt)
        actual_simtime = n_steps * self.dt
        if rebuild_simulator:
            connlist = ConnectionList(self.connections.values())
            self.simulator = connlist.simulator()
        self.simulator.step(self.queue, n_steps, self.dt)
        self.simtime += actual_simtime



