import time
import theano

from ..nef_theano.api2 import (
    LIFNeuron,
    random_low_rank_connection,
    Simulator
    )

#from .. import nef_theano as nef
for n_ensembles in [10]:#, 100, 1000]:
    for size in [10, 100, 1000]:
        for rank in [1, 2, 50]:
            simtime = 0.5
            dt = 0.001

            p = LIFNeuron(size=size * n_ensembles)
            pops = [p[ii * size:(ii + 1) * size]
                for ii in range(n_ensembles)]
            connections = [random_low_rank_connection(p1, p2, rank)
                for p1, p2 in zip(pops[:-1], pops[1:])]


            #Filter(.001, .03, source=D.output)

            sim = Simulator([p], connections)
            #print '-' * 80
            #theano.printing.debugprint(sim.f)
            n_steps = 100
            t0 = time.time()
            sim.step(n_steps)
            t1 = time.time()
            #print "runtime: ", (t1 - t0), "seconds"
            print n_ensembles, size, rank,
            print 'speed:', (n_steps / (t1 - t0)), 'steps/second'


