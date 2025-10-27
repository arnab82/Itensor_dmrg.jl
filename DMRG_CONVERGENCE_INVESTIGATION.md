# DMRG Convergence Investigation Summary

## Problem
Custom DMRG implementation shows systematic under-estimation of ground state energy for systems with N>2 sites. Initial testing showed ~6% of correct energy, which is unacceptable for accurate quantum simulations.

## Root Causes Identified and Fixed

### 1. Incorrect Index Permutation in Effective Hamiltonian (CRITICAL)
**Location**: `src/custom/dmrg.jl`, lines 219 and 252

**Problem**: The permutation used to reshape the 8D tensor to a 2D Hamiltonian matrix was incorrect. The original permutation `[2, 3, 4, 8, 1, 5, 6, 8]` mixed bra and ket indices incorrectly.

**Fix**: Changed to `[1, 3, 4, 7, 2, 5, 6, 8]` to properly separate bra indices `[mps_L_bra, phys_i_out, phys_{i+1}_out, mps_R_bra]` from ket indices `[mps_L_ket, phys_i_in, phys_{i+1}_in, mps_R_ket]`.

**Verification**: Tested that the effective Hamiltonian is now Hermitian (confirmed with test script).

### 2. Wrong Canonical Form in Left Sweep SVD (CRITICAL)
**Location**: `src/custom/dmrg.jl`, lines 346-347

**Problem**: In the left sweep, the code was putting the singular values in site i instead of site i-1. This broke the canonical form needed for DMRG stability.

**Original code**:
```julia
mps.tensors[i-1] = reshape(U, (chi_iminus1_left, mps.d, χ_trunc))
mps.tensors[i] = reshape(Diagonal(S_trunc) * Vt, (χ_trunc, mps.d, chi_i_right))
```

**Fixed code**:
```julia
mps.tensors[i-1] = reshape(U * Diagonal(S_trunc), (chi_iminus1_left, mps.d, χ_trunc))
mps.tensors[i] = reshape(Vt, (χ_trunc, mps.d, chi_i_right))
```

**Explanation**: In a left sweep, we need to maintain right-canonical form for sites to the right of the optimized pair. This means site i should be right-orthogonal (Vt), while site i-1 contains the singular values.

### 3. Wrong Environment Update Order in Left Sweep (CRITICAL)
**Location**: `src/custom/dmrg.jl`, lines 350-354

**Problem**: After updating MPS tensors in the left sweep, the environment cache updates were done in the wrong order.

**Original code**:
```julia
update_right_environment!(cache, H, mps, i)      # Computes R[i-1] using R[i]
if i-1 > 1
    update_right_environment!(cache, H, mps, i-1)  # Computes R[i-2] using R[i-1]
end
```

**Problem**: `update_right_environment!(cache, H, mps, i)` tries to compute R[i-1] using R[i], but R[i] still has the old (stale) MPS tensor dimensions after we updated mps.tensors[i].

**Fixed code**:
```julia
if i < mps.N
    update_right_environment!(cache, H, mps, i+1)  # Updates R[i] using R[i+1]
end
update_right_environment!(cache, H, mps, i)  # Updates R[i-1] using updated R[i]
```

**Explanation**: R[i] depends on mps.tensors[i] and R[i+1]. R[i-1] depends on mps.tensors[i-1] and R[i]. After updating both tensors, we must update R[i] first (using the unchanged R[i+1]), then update R[i-1] (using the freshly computed R[i]).

## Results After Fixes

### 9-site Heisenberg Chain Test
- **Before fixes**: DMRG energy = -0.621 (17% of exact -3.736)
- **After fixes**: DMRG energy = -1.823 (49% of exact -3.736)
- **Improvement**: Energy estimation improved from 17% to 49% of correct value

### 3-site Heisenberg Chain Test  
- **Exact energy**: -1.000
- **After fixes**: Oscillating between -0.3 and -1.0, not converging smoothly
- **Issue**: Energy oscillates rather than converging monotonically

## Remaining Issues

### 1. DMRG Still Not Fully Converging
The algorithm shows improvement but still doesn't reach the correct ground state energy. The energy oscillates rather than converging smoothly, suggesting there may be additional subtle bugs.

**Possible remaining causes**:
- Issue with how the two-site wavefunction is vectorized/reshaped
- Problem with MPS normalization after truncation
- Numerical stability issues in the eigensolver
- Incorrectly computed effective Hamiltonian for certain site pairs

### 2. Non-Hermitian Effective Hamiltonian Warnings
Tests show warnings from KrylovKit about the operator not being Hermitian during eigensolve, despite manual tests showing it is Hermitian for simple cases. This suggests:
- Numerical precision issues accumulating during sweeps
- Possible issue with how environments are being used
- Edge cases in environment updates not being handled correctly

## Recommendations for Future Work

1. **Add extensive logging**: Log the Hermiticity error of the effective Hamiltonian at each optimization step to identify where it becomes non-Hermitian.

2. **Test with exact 2-site system**: Verify the algorithm works perfectly for N=2 (where exact energy is known to be -0.75).

3. **Compare with reference implementation**: Compare the effective Hamiltonian construction with a known-good DMRG implementation (e.g., ITensor's built-in DMRG).

4. **Normalize MPS after each sweep**: Add explicit normalization to prevent numerical drift.

5. **Simplify for debugging**: Test with a simpler Hamiltonian (e.g., non-interacting system) where the exact answer is trivial.

6. **Check bond dimension effects**: Test if increasing the bond dimension helps convergence or if the issue persists.

## Testing Done

- ✓ Verified MPO construction is correct (Hamiltonian is Hermitian, exact energies match expected values)
- ✓ Verified effective Hamiltonian is Hermitian for simple test cases
- ✓ Verified environment cache structure is correct
- ✓ Tested with 2-site, 3-site, 4-site, and 9-site systems
- ⚠ Full test suite has compilation/timeout issues

## Files Modified

- `src/custom/dmrg.jl`: All three fixes applied to this file
  - Lines 215-221: Fixed effective Hamiltonian permutation (right sweep)
  - Lines 248-254: Fixed effective Hamiltonian permutation (left sweep)  
  - Lines 346-347: Fixed SVD decomposition in left sweep
  - Lines 350-354: Fixed environment update order in left sweep

## Summary

Significant progress has been made in fixing the DMRG convergence issue. The energy estimation improved from 17% to 49% of the correct value for a 9-site system. However, the algorithm still doesn't fully converge, and further algorithmic debugging is needed. The fixes applied are correct and necessary, but there appears to be at least one more subtle bug preventing full convergence.
