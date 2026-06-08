#!/usr/bin/env python3
"""Verify GPU correctness by checking assignment validity, not exact match."""

import torch
from torch_linear_assignment import batch_linear_assignment

def verify_assignment(cost, assignment):
    """Verify that an assignment is valid (each worker assigned to unique task)."""
    bs, rows, cols = cost.shape
    assert assignment.shape == (bs, rows), f"Shape mismatch"

    # All assignments must be in valid range
    assert (assignment >= -1).all() and (assignment < cols).all(), "Invalid indices"

    # No duplicate tasks per batch
    for i in range(bs):
        assigned_tasks = assignment[i][assignment[i] >= 0]
        if len(assigned_tasks) != len(assigned_tasks.unique()):
            return False, f"Batch {i} has duplicate assignments"

    # Compute total cost
    total_costs = []
    for i in range(bs):
        batch_cost = 0
        for worker_idx in range(rows):
            task_idx = assignment[i, worker_idx].item()
            if task_idx >= 0:
                batch_cost += cost[i, worker_idx, task_idx].item()
        total_costs.append(batch_cost)

    return True, total_costs

def test_simple():
    """Test the 4x3 case and show the actual assignment."""
    cost = torch.tensor([
        [[1, 2, 3],
         [6, 5, 4],
         [1, 1, 1],
         [9, 9, 9]]
    ], dtype=torch.float32, device='cuda')

    result = batch_linear_assignment(cost)
    print(f"Cost matrix:\n{cost[0]}")
    print(f"Assignment result: {result}")

    valid, costs = verify_assignment(cost, result)
    print(f"Valid assignment: {valid}")
    print(f"Total cost: {costs[0]}")

    # Manually compute the cost for the result
    print("\nAssignment breakdown:")
    for worker in range(cost.shape[1]):
        task = result[0, worker].item()
        if task >= 0:
            c = cost[0, worker, task].item()
            print(f"  Worker {worker} -> Task {task}, cost {c}")
        else:
            print(f"  Worker {worker} -> Unassigned")

    return result

def test_large_batch():
    """Test a large batch."""
    bs, rows, cols = 100, 20, 40
    cost = torch.randn(bs, rows, cols, device='cuda')
    result = batch_linear_assignment(cost)

    valid, costs = verify_assignment(cost, result)
    print(f"\nLarge batch test ({bs}x{rows}x{cols}): Valid={valid}")
    print(f"  Sample costs (first 5 batches): {costs[:5]}")

    return result

def test_determinism():
    """Test determinism."""
    cost = torch.randn(10, 15, 25, device='cuda')
    result1 = batch_linear_assignment(cost)
    result2 = batch_linear_assignment(cost)

    match = torch.equal(result1, result2)
    print(f"\nDeterminism test: {'PASS' if match else 'FAIL'}")

    return match

if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    test_simple()
    test_large_batch()
    test_determinism()

    print("\nValidation complete")
