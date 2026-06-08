#!/usr/bin/env python3
"""Test 3: Large recurrent network stress test on gfx1100"""
from brian2 import *
import brian2cuda
import os

os.environ['USE_HIP'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

print("=== Test 3: Large Synapse Network (Stress Test) ===")
set_device('cuda_standalone', build_on_run=False)

N = 200
tau = 10*ms

G = NeuronGroup(N, 'dv/dt = -v/tau : 1', threshold='v > 1', reset='v = 0', method='linear')
G.v = 'rand()'

S = Synapses(G, G, on_pre='v += 0.1')
S.connect(p=0.3)

spikemon = SpikeMonitor(G)

run(20*ms)
device.build(directory='/tmp/brian2_test3', compile=True, run=True, clean=False)

n_synapses = len(S.i)
n_spikes = len(spikemon.t)
avg_spikes = n_spikes / N if N > 0 else 0

print(f"  Neurons: {N}")
print(f"  Synaptic connections: {n_synapses}")
print(f"  Total spikes: {n_spikes}")
print(f"  Avg spikes per neuron: {avg_spikes:.1f}")
print(f"  PASS: Large network simulation completed without deadlock on wave32")
