# 25-Site DMRG Guide

## Overview

This document provides guidance on running DMRG calculations for 25-site systems using the custom DMRG implementation.

## Quick Answer

**YES**, the custom DMRG code CAN handle 25-site calculations. Use **two-site DMRG** for best results.

## Usage

### Basic 25-Site Calculation

```julia
include("src/custom/custom_dmrg.jl")

# System parameters
N = 25
d = 2
chi_mps = 20

# Create Hamiltonian
H = heisenberg_ham(N, d, 5)

# Initialize MPS
mps = random_MPS(N, d, chi_mps)

# Run two-site DMRG
energy, mps = dmrg(H, mps, 10, chi_mps, 1e-8, false)
```

See `example/25_site_dmrg_demo.jl` for a complete working example.

## Performance Tips

### 1. Bond Dimension
- Start with `chi = 20-30` for 25 sites
- Increase if convergence is poor
- Larger chi = better accuracy but slower

### 2. Convergence
- Use 10-20 sweeps for initial optimization
- Monitor energy convergence
- May need more sweeps for larger systems

### 3. Initial State
- Random initialization works but is slow
- Better: use result from smaller system as starting point
- Or use product state close to expected ground state

## Known Issues

### Single-Site DMRG
**DO NOT USE** single-site DMRG in the current implementation. It has bugs that cause:
- Near-zero energies
- Loss of optimization progress
- Incorrect results

**Stick to two-site DMRG** for reliable results.

## Optimization Strategies

### For Faster Calculations

1. **Reduce frequency of environment rebuilds**: The current implementation rebuilds environments before each sweep, which is conservative but slow.

2. **Use incremental updates**: Enable incremental environment updates during sweeps (currently disabled in two-site DMRG).

3. **Adjust truncation threshold**: The `svd_threshold` in two-site DMRG can be tuned (currently `1e-14`).

### For Better Convergence

1. **Increase bond dimension gradually**: Start with small chi, increase over sweeps.

2. **Use tighter tolerance**: Set `tol = 1e-10` or smaller for better convergence.

3. **More sweeps**: 20-50 sweeps may be needed for difficult systems.

## Benchmarks

Typical performance on 25-site Heisenberg model (chi=20):
- Time per sweep: ~10-20 seconds
- Total time (10 sweeps): ~2-3 minutes
- Memory usage: Moderate

## Troubleshooting

### Convergence Issues
- Energy oscillates: Increase number of sweeps
- Energy doesn't improve: Try larger bond dimension
- Too slow: Reduce bond dimension or number of sweeps

### Memory Issues
- Reduce bond dimension (chi)
- Use fewer sweeps initially
- Process smaller systems first

## Alternative: ITensor Built-in DMRG

For production calculations, consider using ITensor's built-in DMRG which is more mature and optimized:

```julia
using ITensors
using ITensorMPS

sites = siteinds("S=1/2", 25)
H = MPO(heisenberg_hamiltonian(4, 7, 1.0), sites)  # Example
ψ0 = randomMPS(sites)
sweeps = Sweeps(10)
setmaxdim!(sweeps, 20, 40, 80)
setcutoff!(sweeps, 1E-8)
energy, ψ = dmrg(H, ψ0, sweeps)
```

## Summary

- ✅ **25-site two-site DMRG works**
- ❌ **Single-site DMRG has bugs**  
- 💡 **Use bond dimension 20-30 for good balance**
- ⏱️ **Expect 2-5 minutes for 10 sweeps**
