#!/usr/bin/env python3
"""Verify the assignment is optimal by enumerating all possibilities."""

import torch
from itertools import permutations

cost = torch.tensor([
    [[1, 2, 3],
     [6, 5, 4],
     [1, 1, 1],
     [9, 9, 9]]
], dtype=torch.float32)

# We have 4 workers and 3 tasks, so max 3 workers can be assigned
# Try all possible assignments of 3 workers to 3 tasks
best_cost = float('inf')
best_assignment = None

# Choose 3 workers from 4
from itertools import combinations
for workers in combinations(range(4), 3):
    # Permute tasks for these 3 workers
    for tasks in permutations(range(3)):
        total_cost = sum(cost[0, w, t].item() for w, t in zip(workers, tasks))
        if total_cost < best_cost:
            best_cost = total_cost
            best_assignment = (workers, tasks)
            print(f"New best: workers {workers} -> tasks {tasks}, cost {total_cost}")

print(f"\nOptimal cost: {best_cost}")
print(f"Optimal assignment: {best_assignment}")

# GPU result was: [0, 2, 1, -1]
# Worker 0 -> Task 0, cost 1
# Worker 1 -> Task 2, cost 4
# Worker 2 -> Task 1, cost 1
# Total: 6
gpu_cost = cost[0, 0, 0] + cost[0, 1, 2] + cost[0, 2, 1]
print(f"\nGPU result cost: {gpu_cost.item()}")
print(f"Match optimal: {gpu_cost.item() == best_cost}")
