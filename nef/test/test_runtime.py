"""This test file is for checking the run time of the theano code."""

import math
import time

from .. import nef_theano as nef

net=nef.Network('Runtime Test')
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

stepsize = 0.002
timesteps = 5000
timesteps = 50

print "making warmup call"
t0 = time.time()
net.run(stepsize)
t1 = time.time()
print "... warmup call took", (t1 - t0), 'seconds'

import theano
theano.printing.debugprint(net.workspace.compiled_updates['step'].ufgraph.fgraph.outputs)

print "Running remaining %s timesteps..." % (timesteps - 1)
start_time = time.time()
for i in range(timesteps - 1):
    net.run(stepsize)
end_time = time.time()
print "... done"
print "... Runtime: ", end_time - start_time, "seconds"
print "... Average: %f steps/second" % (
        timesteps / (end_time - start_time),)

