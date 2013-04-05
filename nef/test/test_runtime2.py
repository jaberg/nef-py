"""This test file is for checking the run time of the theano code."""

import math
import sys
import time

from .. import nef_theano as nef

import theano
from theano_workspace import optimize_methods
from theano_workspace.workspace import ViewWorkspace
import theano_workspace.opt as opt

net=nef.Network('Runtime Test', seed=123)
net.make_input('in', value=math.sin)
net.make('A', 1000, 1)
net.make('B', 1000, 1)
net.make('C', 1000, 1)
net.make('D', 1000, 1)

# some functions to use in our network
def pow(x):
    return [xval**2 for xval in x]

def mult(x):
    return [xval*2 for xval in x]

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('A', 'C', func=pow)
net.connect('A', 'D', func=mult)
net.connect('D', 'B', func=pow) # throw in some recurrency whynot


print "Making theano_tick"
net.make_theano_tick()
net.workspace = optimize_methods(
    ViewWorkspace(net.workspace),
    theano.compile.mode.OPT_STABILIZE)
#net.workspace = optimize(self.workspace)

print '... done'
if 1:
    import theano
    fgraph = net.workspace.compiled_updates['step'].ufgraph.fgraph
    print 'last pass'
    opt.rst.apply(fgraph)
    theano.printing.debugprint(fgraph.outputs, file=open('fgraph.txt', 'wb'))
    print 'N ELEMS', len(fgraph.toposort())
if 0:
    from theano_workspace import profiling
    profs = profiling.add_profilers(net.workspace)
    print net.workspace.step
    assert net.workspace.step.profiler
    simtime = 0.1
else:
    profs = None
    simtime = 0.5

print "Warmup for 0.003 seconds"
t0 = time.time()
net.run(.003)
t1 = time.time()
print "... warmup call took", (t1 - t0), 'seconds'


print "Timing simulation of %s seconds..." % simtime
start_time = time.time()
net.run(simtime)
end_time = time.time()
print "... done"
print "... Runtime: ", end_time - start_time, "seconds"
if profs:
    n_calls = profs['step'].fct_callcount
else:
    n_calls = simtime / net.dt
print "... Average: %f steps/second" % (
        n_calls / (end_time - start_time),)

if profs:
    profs['step'].summary(sys.stdout)


if 0:
    if 1:
      for node in fgraph.toposort():
        if any([('float64' in str(v.dtype)) for v in node.outputs]):
            print 'HAS 64bit output', node.op
            theano.printing.debugprint(node.outputs)
            break

        if 0 and 'dot' in str(node.op):
            print 'Dot:'
            print node.op
            print node.inputs
            print [v.type for v in node.inputs]
            theano.printing.debugprint(node.outputs)
            if any([('64' in str(v.dtype)) for v in node.inputs]):
                print 'HAS 64bit input', node.op
            break

