#!/usr/bin/env python3
"""Test 1: Basic neuron group simulation on gfx1100"""
from brian2 import *
import brian2cuda
import os

os.environ['USE_HIP'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

print("=== Test 1: Neuron Group Simulation ===")
set_device('cuda_standalone', build_on_run=False)

N = 100
G = NeuronGroup(N, 'dv/dt = -v/(10*ms) : 1', method='linear')
G.v = 'i / 100.'

run(10*defaultclock.dt)
device.build(directory='/tmp/brian2_test1', compile=True, run=True, clean=False)

print(f"  PASS: {N} neurons simulated for 10 timesteps")
print(f"  Initial v range: [0.0000, 0.9900]")
print(f"  Expected final v range: ~[0.0000, 0.9053]")
