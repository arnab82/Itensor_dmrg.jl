# Custom DMRG Analysis Summary

## Current Status

The custom DMRG implementation has been improved but still does not fully converge for N≥3 sites.

### What Works
- ✅ 2-site Heisenberg chain: **PERFECT** (-0.75 exact vs -0.75 DMRG)
- ✅ Hamiltonian MPO construction: Correct (verified against exact diagonalization)
- ✅ Effective Hamiltonian construction: Hermitian and gives correct eigenvalue (-1.0) at each optimization
- ✅ SVD decomposition: Exact reconstruction (error ~1e-16)
- ✅ Environment caching infrastructure: Working as designed

### What Doesn't Work
- ❌ 3-site Heisenberg chain: Converges to wrong energy
  - With cache reinitialization: Oscillates between -0.5 and -0.95 (exact: -1.0)
  - Without cache reinitialization: Converges smoothly to ~-0.43 to -0.91 (40-60% error)

## Key Findings

### 1. Effective Hamiltonian is Correct
Each eigensolve gives the correct ground state energy (-1.0) for the effective Hamiltonian. This means:
- The environment construction is correct
- The effective Hamiltonian tensor contractions are correct  
- The permutation indices are correct

### 2. The Problem Occurs After SVD
After optimizing sites (i, i+1):
- Eigensolve energy: -1.0 ✅
- MPS energy after SVD and update: -0.4 to -0.9 ❌

This suggests the issue is in how the two-site ground state is decomposed back into individual MPS tensors, OR in how the full system energy is computed.

### 3. Incremental Environment Updates are Essential
Without incremental updates during sweeps, the effective Hamiltonian uses stale environments and gives wrong results. The current implementation includes:
- **Right sweep**: Updates L[i] and L[i+1] after optimizing (i, i+1)
- **Left sweep**: Updates R[i] and R[i-1] after optimizing (i-1, i)

### 4. Cache Reinitialization Strategy Matters
Different strategies give different behaviors:
- **Full reinit before each half-sweep**: Causes oscillations but reaches higher energies (-0.95)
- **Reinit only at start**: Smooth convergence but to wrong value (-0.43 to -0.91)
- **Partial reinit (R only before left sweep)**: Intermediate behavior

## Implemented Fixes

All fixes documented in previous commits have been applied:

1. ✅ **Correct permutation in effective Hamiltonian** ([1, 3, 4, 7, 2, 5, 6, 8])
2. ✅ **Correct SVD canonical form in left sweep** (S in site i-1)
3. ✅ **Correct environment update order** (R[i] then R[i-1] in left sweep)
4. ✅ **Comprehensive environment updates** (Both L[i] and L[i+1] in right sweep)

## Possible Root Causes

### Most Likely
1. **Mismatch between effective H and full H**: The effective Hamiltonian optimizes a subset while keeping other sites fixed. The way this is stitched back together may lose optimality.

2. **Truncation in 2-site vs 3-site**: For 2 sites, there's no middle bond to truncate. For 3 sites, bonds (1-2) and (2-3) both get truncated, possibly losing critical entanglement.

3. **Bond dimension limitation**: Even though χ_max=20 should be sufficient for a 3-site d=2 system (Hilbert space = 8), maybe the variational space is restricted.

### Less Likely  
4. **Subtle indexing bug**: Could be in tensor reshaping, environment construction, or somewhere else.

5. **Numerical precision issues**: Accumulating errors in environment computations.

## Recommendations

### Immediate Next Steps
1. **Compare with reference implementation**: Study ITensor's DMRG source code in detail to identify algorithmic differences.

2. **Add extensive debugging**: Log:
   - Energy after each bond optimization
   - Singular value spectrum at each SVD
   - Entanglement entropy
   - Norm of MPS at each step
   - Hermiticity error of effective H during sweeps

3. **Test with different Hamiltonians**: Try non-interacting systems where exact answer is trivial.

4. **Single-site DMRG**: Implement single-site DMRG as comparison (more stable, less variational freedom).

### Longer Term
5. **Systematic unit tests**: Test each component (environments, contractions, SVD, etc.) in isolation.

6. **Profile numerical stability**: Check condition numbers of matrices, singular value decay.

7. **Consider alternative algorithms**: DMRG variants like zip-up DMRG, or other tensor network methods.

## Test Commands

```bash
# 2-site test (works perfectly)
julia --project=. test_2_site_debug.jl

# 3-site test (shows the problem)
julia --project=. test_3_site_debug.jl

# ITensor reference (for comparison)
julia --project=. -e 'using ITensors, ITensorMPS; ...'
```

## Conclusion

The custom DMRG implementation is close but has a fundamental issue preventing convergence for N≥3. All documented fixes from the previous attempt have been applied. The problem appears to be in how the variational optimization across multiple bonds is coordinated, rather than in any single component. Further investigation requires either:
- Deep comparison with a working reference implementation
- Extensive debugging to trace where optimality is lost
- Expert consultation on DMRG algorithm details
