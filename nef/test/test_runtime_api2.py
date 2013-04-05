import time

from ..nef_theano.api2 import (
    LIFNeuron,
    RandomLowRankConnection,
    RandomLowRankConnection,
    Simulator
    )



p = LIFNeuron(size=4000)

connections = []
A = p[:1000]
B = p[1000:2000]
C = p[2000:3000]
D = p[3000:]
#connections.append(NEFConnection(A, B, 1, value=sin))
connections.append(RandomLowRankConnection(A, B, 1))
connections.append(RandomLowRankConnection(A, C, 1))
connections.append(RandomLowRankConnection(A, D, 1))
connections.append(RandomLowRankConnection(D, B, 1))

#Filter(.001, .03, source=D.output)


sim = Simulator([p], connections)
n_steps = 100
print "starting simulation"
t0 = time.time()
sim.step(n_steps)
t1 = time.time()
print "runtime: ", (t1 - t0), "seconds"
print 'speed:', (n_steps / (t1 - t0)), 'steps/second'


