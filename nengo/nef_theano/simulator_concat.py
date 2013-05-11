"""
simulator_concat: a theano simulator that merges together
populations and certain other ops for faster evaluation.

"""
from _collections import OrderedDict
import theano
import numpy as np

import simulator
from simulator import MapGemv
from simulator import MiscGemv
import lif

from theano.compile import optdb
from theano.gof.opt import Optimizer
from theano.tensor import Subtensor
from theano.tensor import Join
from theano.tensor import inc_subtensor
from theano.tensor import patternbroadcast
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize

@register_canonicalize
@local_optimizer()
def local_join_to_set_subtensors(node):
    if not isinstance(node.op, Join):
        return

    fgraph = node.fgraph
    _shape_of = fgraph.shape_feature.shape_of
    shape_cache = {}
    def shape_of(vv):
        try:
            return shape_cache[vv]
        except KeyError:
            pass
        rval = []
        for ii in range(vv.ndim):
            obj = _shape_of[vv][ii]
            try:
                shpi = theano.tensor.get_scalar_constant_value(obj)
            except theano.tensor.NotScalarConstantError:
                shpi = obj.eval()
            rval.append(shpi)
        rval = map(int, rval)
        shape_cache[vv] = rval
        return rval

    try:
        axis = theano.tensor.get_scalar_constant_value(node.inputs[0])
    except theano.tensor.NotScalarConstantError:
        return

    #lens = [shape_of(vv)[0] for vv in node.inputs[1:]]

    outshape = shape_of(node.outputs[0])

    new_host = theano.tensor.zeros(outshape,
                                   dtype=node.outputs[0].dtype)
    #print new_host.type
    #print node.outputs[0].type

    if axis != 0:
        raise NotImplementedError()
    offset = 0
    for vv in node.inputs[1:]:
        len_i = shape_of(vv)[axis]
        new_host = inc_subtensor(new_host[offset: offset + len_i], 
                                 vv)
        offset += len_i
    if new_host.broadcastable != node.outputs[0].broadcastable:
        new_host = patternbroadcast(new_host, node.outputs[0].broadcastable)

    return [new_host]


# -- support for the "const_shape" tag I made up
def shape_ir(self, i, r):
    """Return symbolic r.shape[i] for tensor variable r, int i"""
    if hasattr(r.type, "broadcastable") and r.type.broadcastable[i]:
        return self.lscalar_one
    elif hasattr(r, 'data'):
        return theano.tensor.as_tensor_variable(r.data.shape[i])
    elif hasattr(r.tag, 'const_shape'):
        return theano.tensor.as_tensor_variable(r.tag.const_shape[i])
    else:
        return theano.tensor.opt.Shape_i(i).make_node(r).outputs[0]

# -- patch this method into the ShapeFeature class
theano.tensor.opt.ShapeFeature.shape_ir = shape_ir


class LIF_Concat(Optimizer):
    """
    Merge multiple populations of LIF neurons
    """
    def apply(self, fgraph):
        lifs = [node for node in fgraph.toposort()
                if isinstance(node.op, lif.LIF_Op)]
        if len(lifs) < 2:
            return

        tau_rcs = [node.op.tau_rc for node in lifs]
        tau_refs = [node.op.tau_ref for node in lifs]
        upsamples = [node.op.upsample for node in lifs]
        dts = [node.inputs[-1] for node in lifs]

        if len(set(tau_rcs)) > 1:
            raise NotImplementedError('multiple tau_rcs')
        if len(set(tau_refs)) > 1:
            raise NotImplementedError('multiple tau_refs')
        if len(set(upsamples)) > 1:
            raise NotImplementedError('multiple upsamples')
        if len(set(dts)) > 1:
            raise NotImplementedError('multiple dts')

        getconst = theano.tensor.get_scalar_constant_value
        shape_of = fgraph.shape_feature.shape_of

        shapes = [[map(int, map(getconst, shape_of[vv]))
                   for vv in node.outputs]
            for node in lifs]

        #sizes = [map(np.prod, shape) for shape in shapes]

        if all(shape[0][0] == 1 for shape in shapes):
            v = theano.tensor.concatenate(
                [node.inputs[0] for node in lifs])

            rt = theano.tensor.concatenate(
                [node.inputs[1] for node in lifs])

            ic = theano.tensor.concatenate(
                [node.inputs[1] for node in lifs])

            dt = lifs[0].inputs[-1]

            new_v, new_rt, new_spiked = lifs[0].op(v, rt, ic, dt)

            repls = []
            for ii, node in enumerate(lifs):
                repls.append((node.outputs[0], new_v[ii:ii + 1]))
                repls.append((node.outputs[1], new_rt[ii:ii + 1]))
                repls.append((node.outputs[2], new_spiked[ii:ii + 1]))
            fgraph.replace_all_validate(repls, reason='LIF_Concat')
        else:
            raise NotImplementedError()


class MapGemv_Concat(Optimizer):
    """
    Merge multiple MapGemv Ops working on subtensors of a single buffer,
    into a MiscGemv that makes potentially irregular accesses to that buffer
    but can potentially do all gemvs in parallel.
    """
    def apply(self, fgraph):
        mgs = [node for node in fgraph.toposort()
               if node.op == MapGemv(False)
               and node.inputs[2].owner
               and isinstance(node.inputs[2].owner.op, Subtensor)
              ]

        if len(mgs) < 2:
            return

        Xs = [node.inputs[2] for node in mgs]

        _shape_of = fgraph.shape_feature.shape_of
        shape_cache = {}
        def shape_of(vv):
            try:
                return shape_cache[vv]
            except KeyError:
                pass
            rval = []
            for ii in range(vv.ndim):
                obj = _shape_of[vv][ii]
                try:
                    shpi = theano.tensor.get_scalar_constant_value(obj)
                except theano.tensor.NotScalarConstantError:
                    shpi = obj.eval()
                rval.append(shpi)
            rval = map(int, rval)
            shape_cache[vv] = rval
            return rval

        X_srcs = [X.owner.inputs[0] for X in Xs]
        #print [X.owner.op.idx_list for X in Xs]
        #print X_srcs
        by_src = {}
        for X_src, node in zip(X_srcs, mgs):
            by_src.setdefault(X_src, []) 
            by_src[X_src].append(node)

        for X_src, mgs_of_X_src in by_src.items():
            if len(mgs_of_X_src) < 2:
                continue
            alphas = [node.inputs[0] for node in mgs_of_X_src]
            As = [node.inputs[1] for node in mgs_of_X_src]
            betas = [node.inputs[3] for node in mgs_of_X_src]
            Ys = [node.inputs[4] for node in mgs_of_X_src]
            if len(set(alphas)) != 1:
                continue
            if len(set(betas)) != 1:
                continue
            alpha = alphas[0]
            beta = betas[0]
            #try:
            A_shp = shape_of(As[0])[1:]
            X_shp = shape_of(Xs[0])[1:]
            Y_shp = shape_of(Ys[0])[1:]
            #except theano.tensor.NotScalarConstantError:
            #    return

            new_Xi_list = []
            new_As_list = []
            new_Ys_list = []
            repl_nodes = []
            # TODO: currently this loop just finds a sequence of Xs
            # that are the 0:1, then 1:2, then 2:3  etc. of X_src.
            # in principle any combination of ranges that makes
            # a contiguous block will do
            for node in mgs_of_X_src:
                #print node
                A = node.inputs[1]
                X = node.inputs[2]
                Y = node.inputs[4]
                if shape_of(A)[1:] != A_shp: continue
                if shape_of(X)[1:] != X_shp: continue
                if shape_of(Y)[1:] != Y_shp: continue
                Xslice = X.owner.op.idx_list[0]
                start = Xslice.start
                if Xslice == slice(start, start + 1, None):
                    new_As_list.append(A)
                    new_Ys_list.append(Y)
                    repl_nodes.append(node)
                    new_Xi_list.append(start)
                    #print "YAY", len(repl_nodes)
            if len(repl_nodes) > 1:
                new_As = theano.tensor.concatenate(new_As_list)
                new_Xs = X_src
                new_Xi = theano.tensor.as_tensor_variable(new_Xi_list)
                new_Ys = theano.tensor.concatenate(new_Ys_list)

                new_out = MiscGemv(False)(alpha, new_As, new_Xs, new_Xi,
                                     beta, new_Ys)

                repl = []
                for ii, node in enumerate(repl_nodes):
                    old = node.outputs[0]
                    new = new_out[ii:ii + 1]
                    # -- N.B. only do this if the new_out is guaranteed
                    #    to have leading dim of 1
                    new = theano.tensor.addbroadcast(new, 0)
                    if old.type != new.type:
                        print shape_of(old)
                        print new_out.broadcastable
                        print old.type, new.type, new.broadcastable, ii, node
                    repl.append((old, new))
                fgraph.replace_all_validate(repl, reason="MapGemv_Concat")


optdb['canonicalize'].register('LIF_Concat', LIF_Concat(),
                               'fast_run', 'nengo')
optdb['canonicalize'].register('MapGemv_Concat', MapGemv_Concat(),
                               'fast_run', 'nengo')



class SimulatorConcat(simulator.Simulator):
    def __init__(self, network):
        self.network = network
        self.simulation_steps = 0

        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')

        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        # for every node in the network
        for node in self.network.nodes.values():
            # if there is some variable to update
            if hasattr(node, 'update'):
                # add it to the list of variables to update every time step
                updates.update(node.update(self.network.dt))

        # create graph and return optimized update function
        self.step = theano.function([simulator.simulation_time], [],
                                    updates=updates.items())


    def run_steps(self, N):
        fn = self.ifs.fn
        for i in xrange(N):
            simulation_time = self.simulation_steps * self.network.dt
            fn(simulation_time)
            self.simulation_steps += 1


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)


