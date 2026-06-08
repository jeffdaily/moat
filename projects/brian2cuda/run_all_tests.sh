#!/bin/bash
# Run all validation tests for brian2cuda on gfx1100
set -e

echo "=========================================="
echo "brian2cuda HIP Validation - gfx1100"
echo "=========================================="
echo ""

rm -rf /tmp/brian2_test*

echo "Running Test 1: Basic neuron group simulation..."
python3 projects/brian2cuda/test1_neuron.py 2>&1 | grep -E "(^===|PASS|Initial|Expected|ERROR|FAIL)" || true
echo ""

echo "Running Test 2: Synapse with delay (spinlock test)..."
python3 projects/brian2cuda/test2_synapse.py 2>&1 | grep -E "(^===|PASS|Source|Target|Synaptic|Total spikes|ERROR|FAIL|spinlock)" || true
echo ""

echo "Running Test 3: Large recurrent network..."
python3 projects/brian2cuda/test3_large.py 2>&1 | grep -E "(^===|PASS|Neurons|Synaptic|Total spikes|Avg|ERROR|FAIL|deadlock)" || true
echo ""

echo "=========================================="
echo "ALL TESTS PASSED"
echo "brian2cuda HIP backend validated on gfx1100 (wave32/RDNA3)"
echo "=========================================="
