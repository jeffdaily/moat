#!/usr/bin/env python3
"""Comprehensive GPU validation test for gfx1100."""

import torch
from torch_linear_assignment import batch_linear_assignment

def test_simple_deterministic():
    """Test 1: Simple deterministic case (4x3 matrix)."""
    cost = torch.tensor([
        [[1, 2, 3],
         [6, 5, 4],
         [1, 1, 1],
         [9, 9, 9]]
    ], dtype=torch.float32, device='cuda')

    result = batch_linear_assignment(cost)
    expected = torch.tensor([[0, 2, -1, 1]], device='cuda')

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("Test 1 PASSED: Simple deterministic 4x3 assignment")
    return result

def test_large_batch():
    """Test 2: Large batch to stress GPU kernel."""
    bs, rows, cols = 100, 20, 40
    cost = torch.randn(bs, rows, cols, device='cuda')
    result = batch_linear_assignment(cost)

    # Verify shape
    assert result.shape == (bs, rows), f"Shape mismatch: {result.shape} != {(bs, rows)}"

    # Verify each batch has valid assignments (indices in [0, cols-1] or -1)
    assert (result >= -1).all() and (result < cols).all(), "Invalid task indices found"

    # Verify no duplicates within each batch
    for i in range(bs):
        assigned = result[i][result[i] >= 0]
        assert len(assigned) == len(assigned.unique()), f"Batch {i} has duplicate assignments"

    print(f"Test 2 PASSED: Large batch {bs}x{rows}x{cols} - all assignments valid, no duplicates")
    return result

def test_empty_batch():
    """Test 3: Empty batch (0x10x10)."""
    cost = torch.randn(0, 10, 10, device='cuda')
    result = batch_linear_assignment(cost)

    assert result.shape == (0, 10), f"Empty batch shape mismatch: {result.shape}"
    print("Test 3 PASSED: Empty batch handled correctly")
    return result

def test_run_to_run_determinism():
    """Test 4: Verify determinism (same input -> same output)."""
    cost = torch.randn(10, 15, 25, device='cuda')
    result1 = batch_linear_assignment(cost)
    result2 = batch_linear_assignment(cost)

    assert torch.equal(result1, result2), "Non-deterministic results detected"
    print("Test 4 PASSED: Run-to-run determinism verified")
    return result1

if __name__ == '__main__':
    print(f"GPU device: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    test_simple_deterministic()
    test_large_batch()
    test_empty_batch()
    test_run_to_run_determinism()

    print("\nAll GPU validation tests PASSED on gfx1100")
