"""

TODO
----
* generate kernels for correct dtypes
* create meta-information dictionary to go with ocl_vars to at least mark constants
* double-buffer the simulator
* use int8 spikes
* use float16 for many things,
"""


from _collections import OrderedDict
import theano
import numpy as np
import pyopencl as cl
import simulator

from ocl.array import to_device

ocl_alloc = {}
ocl_post_alloc_transforms = []
ocl_perform = {}


# -- decorator to register an OCL "perform" method for Theano Op `op_cls`
def perform(op_cls):
    def deco(f):
        ocl_perform[op_cls] = f
        return f
    return deco


# -- decorator to register an OCL "alloc" method for Theano Op `op_cls`
def alloc(op_cls):
    def deco(f):
        ocl_alloc[op_cls] = f
        return f
    return deco


class UnAllocatedOutput(object):
    """Singleton to stand for an un-allocated output """



class IFS(object):
    """
    Iteratable Function System

    Like "FunctionGraph" in Theano, except that it is is designed around
    updates, so the number of inputs and outputs can actually change.
    """
    def __init__(self, fn):
        self.fn = fn
        self.updates = fn.fn.updated_vars.items()
        self.fgraph = fn.maker.fgraph
        for ivar, ovar in self.updates:
            assert ivar.owner is None
        self._nodes = fn.fn.nodes
        self.meta = {}
        for vv in self.variables:
            self.meta.setdefault(vv, copy.copy(vv.tag))
        self.addressable = dict([])

    def add_update(self, in_expr, out_expr):
        if in_expr.owner:
            raise ValueError(in_expr)
        self.updates.append((in_expr, out_expr))

    def replace(self, old_v, new_v):
        raise NotImplementedError()

    @property
    def nodes(self):
        return self._nodes

    @property
    def variables(self):
        """
        Return the variables involved in this IFS.
        """
        tmp = [b for (a, b) in self.updates if b.owner is None]
        for node in self.nodes:
            tmp.extend(node.inputs)
            tmp.extend(node.outputs)
        # name the destinations in case they haven't been used yet
        tmp.extend([a for (a, b) in self.updates])
        seen = set()
        rval = []
        for vv in tmp:
            if vv not in seen:
                rval.append(vv)
            seen.add(vv)
        return rval

def ifs_unalloc(ifs):
    for vv in ifs.variables:
        ifs.meta[vv].ocl0 = UnAllocatedOutput
        ifs.meta[vv].ocl1 = UnAllocatedOutput
        ifs.meta[vv].const_val = UnAllocatedOutput


def ifs_alloc(ifs, queue):
    # allocate storage for the variables in an IFS
    # by copying things from Theano and by calling the
    # allocators in ocl_alloc
    for node in ifs.nodes:
        for vv in node.inputs:
            if not (ifs.meta[vv].ocl0 is UnAllocatedOutput
                and ifs.meta[vv].const_val is UnAllocatedOutput):
                continue
            if hasattr(vv, 'data'):
                ifs.meta[vv].const_val = vv.data
            elif vv.owner is None:
                val = vv.get_value(borrow=True)
                ifs.meta[vv].ocl0 = to_device(queue, val)
        ocl_alloc[type(node.op)](queue, ifs, node)
        for vout in node.outputs:
            assert (ifs.meta[vout].ocl0 is not UnAllocatedOutput
                or ifs.meta[vout].const_val is not UnAllocatedOutput)
            if ifs.meta[vout].ocl0 is not UnAllocatedOutput:
                assert ifs.meta[vout].ocl0.__class__.__name__ == 'Array', (
                    node.op)
                assert ifs.meta[vout].ocl0.ndim == vout.ndim, node.op
                assert ifs.meta[vout].ocl0.dtype == vout.dtype, node.op
            else:
                assert ifs.meta[vout].const_val is not None
    # -- the design is that by the time we allocate buffers
    #    we have guaranteed that the updates are implemented
    #    by destructive copy operations (tensor.second) so
    #    the outputs naturally go into the inputs for the next
    #    round of computation.
    queue.finish()



def ifs_arrays(ifs, varlist):
    return [ifs.meta[vv].ocl0 for vv in varlist]


def ifs_consts(ifs, varlist):
    return [ifs.meta[vv].const_val for vv in varlist]

def ifs_set_arrays(ifs, varlist, vallist):
    assert len(varlist) == len(vallist)
    for vv, vl in zip(varlist, vallist):
        ifs.meta[vv].ocl0 = vl

def ifs_set_consts(ifs, varlist, vallist):
    assert len(varlist) == len(vallist)
    for vv, vl in zip(varlist, vallist):
        ifs.meta[vv].const_val = vl

def ifs_from_network(network):
    # dictionary for all variables
    # and the theano description of how to compute them 
    updates = OrderedDict()

    # for every node in the network
    for node in network.nodes.values():
        # if there is some variable to update
        if hasattr(node, 'update'):
            # add it to the list of variables to update every time step
            updates.update(node.update(network.dt))

    # make every update to be one of these guys so that
    # (a) every variable is actually involved in computations and
    # (b) we know how to make the graph represent an iterative computation
    #     namely by forcing these elemwise ops to be in-place before
    #     considering any other destructive operations
    for ivar, ovar in updates.items():
        updates[ivar] = theano.tensor.second(ivar, ovar)

    # create graph and return optimized update function
    # -- use py linker to avoid wasting time compiling C code
    # -- don't insert device specializations
    # -- don't insert destructive operations
    OPT = theano.gof.Query(include=['fast_run'], exclude=[])
    OPT.position_cutoff = 20.0
    updates_items = updates.items()
    fn = theano.function([simulation_time], [],
        updates=updates_items, 
        mode=theano.Mode(
            optimizer=OPT,
            linker=theano.gof.vm.VM_Linker(use_cloop=False, allow_gc=False),
            ))
    return IFS(fn)

class SimulatorOCL(object):
    """
    Simulator that uses OpenCL instead of numpy to evaluate the Theano "step"
    function and run the simulator for the network.

    This class draws on alternate implementations for the Ops in the step
    function. It
    """
    def __init__(self, network, context=None, profiling=False):
        self.network = network
        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')
        if context is None:
            context = cl.create_some_context()
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
            self.t_used = {}
        else:
            self.queue = cl.CommandQueue(context)

        self.step = theano_fn_from_network(network)
        ifs = simulator.IFS(self.step)
        self.ifs = ifs

        # -- allocate two plans and vars for double-buffered updates
        self.n_steps = -1  # -- how many steps of dt in total have we done?
        # -- XXX misnomber: n_steps should be last_step
        self.plans = []
        self._node_plans = {}
        self._scribe_buf = {}
        self._simtime = cl.array.zeros(self.queue, (), dtype='float32')

        ifs_unalloc(self.ifs)
        for vv in ifs.variables:
            if vv.name == 'simulation_time':
                ifs.meta[vv].ocl0 = self._simtime
        ifs_alloc(self.ifs, self.queue)

        # -- optimize the ifs graph
        for fn in ocl_post_alloc_transforms:
            fn(self.ifs)

        # -- build plans for evaluating ocl_vals[0]
        for node in self.ifs.nodes:
            plans = ocl_perform[type(node.op)](self.queue, self.ifs, node)
            for plan in plans:
                plan.node = node
            self.plans.extend(plans)
            self._node_plans[node] = plans

        self.queue.finish()

    def copy_to_shared(self):
        """
        Copy data from the theano graph's shared variables into ocl vars
        """

        for (ivar, ovar) in self.ifs.updates:
            try:
                nparray = self.ifs.meta[ivar].ocl0.get(self.queue)
            except AssertionError:
                print self.ifs.meta[ivar].ocl0.structure
                raise
            except AttributeError, e:
                e.args = e.args + (ivar,)
                raise
            ivar.set_value(nparray, borrow=True)
        self.queue.finish()

    def copy_from_shared(self):
        """
        Copy data from ocl vars into the theano graph's shared variables
        """
        for (ivar, ovar) in self.ifs.updates:
            nparray = ivar.get_value(borrow=True)
            assert nparray.dtype == self.ifs.meta[ivar].ocl0.dtype
            assert nparray.shape == tuple(self.ifs.meta[ivar].ocl0.shape)
            self.ifs.meta[ivar].ocl0.set(nparray, self.queue)
        self.queue.finish()

    def run_steps_with_theano(self, N):
        storage_map = self.step.fn.storage_map
        for i in xrange(N):
            self.n_steps += 1
            ocl_vars = self._ocl_vars[self.n_steps % 2]
            node_plans = self._node_plans[self.n_steps % 2]
            for jj, (node, thunk) in enumerate(
                    zip(self.step.fn.nodes, self.step.fn.thunks)):

                def any_inaccuracy(msg, theano_vars):
                    inaccuracy = False
                    seen = set()
                    for ovar in theano_vars:
                        if ovar in seen:
                            continue
                        if ovar in ocl_vars:
                            refval = storage_map[ovar][0]
                            try:
                                oclval = ocl_vars[ovar].get()
                            except (AssertionError, ValueError):
                                print ocl_vars[ovar].structure
                                raise
                            assert refval.dtype == oclval.dtype
                            assert refval.shape == oclval.shape
                            if not np.allclose(refval, oclval,
                                               atol=1e-4, rtol=1e-4):
                                print msg, self.n_steps, 'Node', node, 'messed up', ovar
                                print '  stats', refval.max(), refval.min(), refval.mean(),
                                print 'vs', oclval.max(), oclval.min(), oclval.mean()
                                print '  diff abs', np.max(abs(refval - oclval)),
                                print 'rel', np.max(abs(refval - oclval) / abs(refval + oclval + 1e-12))
                                inaccuracy=True
                        seen.add(ovar)
                    return inaccuracy

                if any_inaccuracy('pre', node.inputs):
                    raise RuntimeError('Inaccurate computations')

                # -- run the theano thunk
                thunk()

                # -- run the ocl equivalent
                for p in node_plans[node]:
                    p._fn(*p._fn_args)
                self.queue.finish()

                vars_that_should_match = node.inputs + node.outputs
                for opos, iposlist in getattr(node.op, 'destroy_map', {}).items():
                    for ipos in reversed(sorted(iposlist)):
                        vars_that_should_match.pop(ipos)
                if any_inaccuracy('post', vars_that_should_match):
                    print node_plans[node]
                    print p._fn
                    print p.text
                    print p._fn_args
                    print vars_that_should_match
                    print [ocl_vars[vv].data for vv in node.inputs if vv in ocl_vars]
                    print [ocl_vars[vv].data for vv in node.outputs]
                    raise RuntimeError('Inaccurate computations')
                else:
                    print '.',
            print 'done pass', self.n_steps
            # -- apply updates to theano fn
            for (ivar, ovar) in self.step.fn.updated_vars.items():
                storage_map[ivar][0] = storage_map[ovar][0]

    def run_steps(self, N, sync_w_theano_shared_vars=True,
                  run_theano_too=False):
        """
        Run the simulator for N steps of duration `self.dt`
        """
        if sync_w_theano_shared_vars and self.n_steps > -1:
            self.copy_from_shared()

        dt = self.network.dt

        if self.profiling:
            for i in xrange(N):
                self.n_steps += 1
                evs = []
                plans = self.plans
                self._simtime.set(
                    np.asarray((self.n_steps + i * 2) * dt, dtype='float32'),
                    queue=self.queue)
                for p in plans:
                    evs.append(p._fn(*p._fn_args))
                self.queue.finish()
                assert len(evs) == len(plans)
                for e, p in zip(evs, plans):
                    self.t_used.setdefault(p.node, 0)
                    self.t_used[p.node] +=  e.profile.end - e.profile.start
        else:
            queues = [p._enqueue_args[0] for p in self.plans]
            kerns = [p._enqueue_args[1] for p in self.plans]
            gsize = [p._enqueue_args[2] for p in self.plans]
            lsize = [p._enqueue_args[3] for p in self.plans]
            try:
                # XXX Only loop for as many times as we have
                # room in the scribe_buf buffers... otherwise
                # the scribe_p kernel will segfault.
                # Then reallocate those buffers, update the pointers
                # of the scribe_p kernel_args, and continue
                #
                # XXX Also update any other kernel_args that are pointing
                # to those buffers.
                for i in xrange(N):
                    simtime = (self.n_steps + i * 2) * dt
                    self._simtime.set(
                        np.asarray(simtime, dtype='float32'),
                        queue=self.queue)
                    map(cl.enqueue_nd_range_kernel,
                        queues, kerns, gsize, lsize)
                self.n_steps += N
            except Exception, e:
                e.args = e.args + ({'plan': p, 'node': p.node},)
                raise

        if sync_w_theano_shared_vars:
            self.copy_to_shared()  # -- calls queue.finish
        else:
            self.queue.finish()

    def run(self, approx_sim_time, **kwargs):
        """
        Run the simulator for a number of steps that advances the total
        simulation time by approximately `approx_sim_time`
        """
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps, **kwargs)


# -- move this import somewhere else?
#    N.B. simulator_ocl (this file) must be imported before ocl_ops
import ocl_ops

