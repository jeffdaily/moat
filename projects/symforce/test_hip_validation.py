#!/usr/bin/env python3
"""
HIP validation test for symforce Caspar backend.

Tests the full code generation pipeline:
1. Create CasparLibrary with symbolic kernels
2. Generate code from Jinja templates
3. Compile with HIP backend for gfx1100
4. Execute kernels on GPU
5. Verify numerical results
"""

import torch
import numpy as np
from pathlib import Path
import symforce.symbolic as sf
from symforce import typing as T
from symforce.caspar import CasparLibrary
from symforce.caspar import memory as mem

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Create a test kernel that exercises multiple Caspar features
caslib = CasparLibrary()

@caslib.add_kernel
def test_kernel(
    a: T.Annotated[sf.V3, mem.ReadShared],
    b: T.Annotated[sf.V3, mem.ReadUnique],
) -> T.Tuple[
    T.Annotated[sf.V2, mem.AddSharedSum],
    T.Annotated[sf.Symbol, mem.WriteIndexed],
]:
    """
    Test kernel that exercises:
    - ReadShared and ReadUnique memory patterns
    - AddSharedSum reduction (uses atomicAdd on shared memory)
    - WriteIndexed scatter
    - Trigonometric operations
    - Arithmetic operations
    """
    # Compute some operations
    sum_result = a + b
    sincos = sf.V2(sf.sin(sum_result[0]), sf.cos(sum_result[1]))
    product = sum_result[2] * 2.0
    return (sincos, product)

# Generate and compile
print("Generating kernel code...")
out_dir = Path("/tmp/symforce_hip_test_gfx1100")
caslib.generate(out_dir)

print("Compiling with HIP for gfx1100...")
caslib.compile(out_dir, use_hip=True, hip_arch='gfx1100')

# Import the compiled library
import sys
sys.path.insert(0, str(out_dir))
import caspar_lib as lib

# Run GPU test
print("Running GPU test...")
device = torch.device('cuda:0')

N = 100
# Create arg0 (ReadShared V3)
arg0_stacked = torch.rand(N, 3, device=device)
arg0_caspar = torch.empty(mem.caspar_size(3), N, device=device)
lib.matrix31_stacked_to_caspar(arg0_stacked, arg0_caspar)

arg0_indices = torch.randint(0, N, (N,), device=device, dtype=torch.int32)
arg0_indices_shared = torch.empty(N, 2, device=device, dtype=torch.int32)
lib.shared_indices(arg0_indices, arg0_indices_shared)

# Create arg1 (ReadUnique V3)
arg1_stacked = torch.rand(1, 3, device=device)
arg1_caspar = torch.empty(mem.caspar_size(3), 1, device=device)
lib.matrix31_stacked_to_caspar(arg1_stacked, arg1_caspar)

# Allocate output0 (AddSharedSum V2)
OUT0_IDX_MAX = 10
out0_caspar = torch.zeros(mem.caspar_size(2), OUT0_IDX_MAX, device=device)
out0_indices = torch.randint(0, OUT0_IDX_MAX, (N,), device=device, dtype=torch.int32)
out0_indices_shared = torch.empty(N, 2, device=device, dtype=torch.int32)
lib.shared_indices(out0_indices, out0_indices_shared)

# Allocate output1 (WriteIndexed Symbol)
out1_caspar = torch.zeros(mem.caspar_size(1), N, device=device)
out1_indices = torch.randperm(N, device=device, dtype=torch.int32)

# Run kernel
lib.test_kernel(
    arg0_caspar,
    arg0_indices_shared,
    arg1_caspar,
    out0_caspar,
    out0_indices_shared,
    out1_caspar,
    out1_indices,
    N,
)

# Convert outputs back to stacked format
out0_stacked = torch.zeros(OUT0_IDX_MAX, 2, device=device)
out1_stacked = torch.empty(N, 1, device=device)
lib.matrix21_caspar_to_stacked(out0_caspar, out0_stacked)
lib.symbol_caspar_to_stacked(out1_caspar, out1_stacked)

# Verify results
print("Verifying results...")
sum_ref = arg0_stacked + arg1_stacked.expand(N, 3)
sincos_ref = torch.stack([torch.sin(sum_ref[:, 0]), torch.cos(sum_ref[:, 1])], dim=1)
product_ref = sum_ref[:, 2] * 2.0

# Check AddSharedSum output
sincos_check = torch.zeros_like(out0_stacked)
for i in range(N):
    sincos_check[out0_indices[i]] += sincos_ref[arg0_indices[i]]
assert torch.allclose(out0_stacked, sincos_check, rtol=1e-4, atol=1e-5), "AddSharedSum output mismatch"

# Check WriteIndexed output
product_check = torch.zeros(N, 1, device=device)
product_check[out1_indices] = product_ref[arg0_indices].unsqueeze(1)
assert torch.allclose(out1_stacked, product_check, rtol=1e-4, atol=1e-5), "WriteIndexed output mismatch"

print("All checks passed!")
print("\n=== VALIDATION PASSED ===")
