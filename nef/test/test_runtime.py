"""This test file is for checking the run time of the theano code."""

import math
import sys
import time

import nef
# some functions to use in our network
def pow(x):
    return [xval**2 for xval in x]

def mult(x):
    return [xval*2 for xval in x]

#from .. import nef_theano as nef
for n_ensembles in [10, 100, 1000]:
    for size in [10, 100]:
        simtime = 0.5
        dt = 0.001

        net=nef.Network('Runtime Test', seed=123)
        #print help(net.make_input)
        net.make_input('in', values=[math.sin])
        for ens in range(n_ensembles):
            net.make(str(ens), size, 1)

        for e1 in range(n_ensembles):
            net.connect('in', str(e1))
            for e2 in range(n_ensembles):
                net.connect(str(e1), str(e2),
                            func=[pow, mult][e1 % 2]
                           )

        print '-' * 80
        print n_ensembles, size
        print "Warmup for 0.003 seconds"
        t0 = time.time()
        net.run(.003, dt)
        t1 = time.time()
        print "... warmup call took", (t1 - t0), 'seconds'


        print "Timing simulation of %s seconds..." % simtime
        start_time = time.time()
        net.run(simtime, dt=dt)
        end_time = time.time()
        print "... done"
        print "... Runtime: ", end_time - start_time, "seconds"
        n_calls = simtime / dt
        print "... Average: %f steps/second" % (
                n_calls / (end_time - start_time),)
