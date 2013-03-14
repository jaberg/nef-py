"""This is a test file to test the noise parameter on ensemble"""

import nef_theano as nef
import numpy as np
import matplotlib.pyplot as plt
import math

net=nef.Network('Noise Test')
net.make_input('in', value=math.sin)
net.make('A', neurons=300, dimensions=1, noise=1)
net.make('A2', neurons=300, dimensions=1, noise=100)
net.make('B', neurons=300, dimensions=2, noise=1000, noise_type='Gaussian')
net.make('C', neurons=100, dimensions=1, array_size=3, noise=10)

net.connect('in', 'A')
net.connect('in', 'A2')
net.connect('in', 'B')
net.connect('in', 'C')

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe(net.nodes['in'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
A2p = net.make_probe(net.nodes['A2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe(net.nodes['C'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ion(); plt.close()
plt.subplot(411); plt.title('Input')
plt.plot(Ip.get_data())
plt.subplot(412); plt.hold(1)
plt.plot(Ap.get_data()); plt.plot(A2p.get_data())
plt.legend(['A noise = 1', 'A2 noise = 100'])
plt.subplot(413); plt.title('B noise = 1000, type = gaussian')
plt.plot(Bp.get_data())
plt.subplot(414); plt.title('C')
plt.plot(Cp.get_data())
