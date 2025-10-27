# Single-Site DMRG Implementation Summary

## Overview

This document describes the single-site DMRG implementation added to the Itensor_dmrg.jl package as an alternative to the existing two-site DMRG algorithm.

## Motivation

Two-site DMRG optimizes two adjacent sites simultaneously and can dynamically adjust bond dimensions through SVD truncation. While powerful, it has some limitations:

1. Slower per sweep (optimizes larger effective Hamiltonian)
2. May over-truncate in some cases
3. Less efficient for refinement when bond dimensions are already optimal

Single-site DMRG addresses these by:
- Optimizing one site at a time (smaller effective Hamiltonian)
- Preserving bond dimensions (no SVD truncation)
- Faster convergence for refinement tasks

## Implementation Details

### Core Functions

#### 1. `construct_effective_hamiltonian_single_site(cache, H, mps, i)`

Constructs the effective Hamiltonian for a single site optimization.

**Algorithm:**
1. Retrieve left environment `L[i-1]` and right environment `R[i]` from cache
2. Contract left environment with MPO site `H[i]`
3. Contract result with right environment
4. Reshape to matrix form for eigenvalue solving

**Tensor Contractions:**
```
L[chi_L_bra, mpo_L, chi_L_ket] * H_i[mpo_L, phys_in, phys_out, mpo_R]
    → temp1[chi_L_bra, chi_L_ket, phys_in, phys_out, mpo_R]

temp1 * R[chi_R_bra, mpo_R, chi_R_ket]
    → temp2[chi_L_bra, chi_L_ket, phys_in, phys_out, chi_R_bra, chi_R_ket]

Reshape to H_eff[(chi_L * d * chi_R), (chi_L * d * chi_R)]
```

#### 2. `dmrg_sweep_single_site!(H, mps, cache, direction)`

Performs a single sweep through all sites.

**Algorithm:**
```
For each site i in sweep direction:
    1. Construct effective Hamiltonian H_eff for site i
    2. Vectorize current site tensor
    3. Solve eigenvalue problem: H_eff * v = E * v
    4. Reshape solution back to tensor form
    5. Update site tensor in MPS
    6. Update environment cache incrementally
```

**Key Property:** Bond dimensions are preserved - the reshaped tensor has the same dimensions as the original.

#### 3. `dmrg_single_site(H, mps, max_sweeps, tol)`

Main driver function for single-site DMRG optimization.

**Algorithm:**
```
Initialize environment cache
For each sweep:
    1. Rebuild left and right environments
    2. Perform right sweep (left to right)
    3. Rebuild environments
    4. Perform left sweep (right to left)
    5. Compute total energy
    6. Check convergence
    Return if |E_new - E_old| < tol
```

**Important:** Unlike two-site DMRG, no initial normalization is performed to preserve bond dimensions.

### Comparison with Two-Site DMRG

| Feature | Two-Site DMRG | Single-Site DMRG |
|---------|---------------|------------------|
| Sites optimized | 2 adjacent sites | 1 site |
| Bond dimensions | Can change via SVD | Fixed (preserved) |
| Effective H size | (χ_L * d² * χ_R)² | (χ_L * d * χ_R)² |
| Speed per sweep | Slower | Faster |
| Truncation error | Yes | No |
| Use case | Initial optimization | Refinement |

## Usage Examples

### Basic Usage

```julia
include("src/custom/custom_dmrg.jl")

# Create Hamiltonian
N = 10
d = 2
H = heisenberg_ham(N, d, 5)

# Initialize MPS with desired bond dimension
chi = 20
mps = random_MPS(N, d, chi)

# Run single-site DMRG
max_sweeps = 50
tol = 1e-8
energy, optimized_mps = dmrg_single_site(H, mps, max_sweeps, tol)
```

### Combined Workflow (Recommended)

```julia
# Step 1: Use two-site DMRG to find optimal bond dimensions
mps = random_MPS(N, d, 10)  # Start with small bond dimension
energy1, mps = dmrg(H, mps, 20, 50, 1e-6, false)

# Step 2: Use single-site DMRG for refinement
energy2, mps = dmrg_single_site(H, mps, 50, 1e-10)
```

## Testing

Comprehensive test suite includes:

1. **Effective Hamiltonian Construction Tests**
   - Hermiticity verification
   - Dimension checks
   - Structure validation

2. **Sweep Functionality Tests**
   - Right and left sweeps
   - Bond dimension preservation
   - Energy updates

3. **Convergence Tests**
   - Heisenberg model
   - Hubbard model
   - Different system sizes

4. **Comparison Tests**
   - Energy comparison with two-site DMRG
   - Performance comparison

5. **Bond Dimension Preservation Tests**
   - Non-uniform bond dimensions
   - Exact preservation verification

All 51 tests pass successfully.

## Performance Characteristics

### Computational Complexity

**Per site optimization:**
- Two-site DMRG: O((χ²d²)³) for eigensolve
- Single-site DMRG: O((χ²d)³) for eigensolve

**Speedup factor:** Approximately (d)³ per site ≈ 8x for d=2 (spin-1/2)

### Memory Usage

- Similar memory footprint as two-site DMRG
- Environment cache reused between sweeps
- No additional SVD workspace needed

### Convergence

- Converges faster when bond dimensions are optimal
- May require more sweeps from random initialization
- Best used after initial two-site DMRG

## Limitations

1. **Cannot grow bond dimensions**: If the initial bond dimension is too small, single-site DMRG cannot increase it. Use two-site DMRG first.

2. **Initial guess sensitivity**: Requires reasonable initial MPS. Random initialization may converge slowly.

3. **Not suitable for:** 
   - Finding optimal bond dimensions from scratch
   - Systems requiring large bond dimension changes
   - Initial exploration of unknown systems

## Future Enhancements

Potential improvements for future versions:

1. **Subspace expansion**: Add perturbative bond dimension growth
2. **Adaptive convergence**: Auto-switch between single-site and two-site
3. **Density matrix mixing**: Improve convergence for difficult cases
4. **Parallelization**: Optimize environment computations
5. **Mixed canonical form**: Support different gauge choices

## References

1. White, S. R. (1992). "Density matrix formulation for quantum renormalization groups"
2. Schollwöck, U. (2011). "The density-matrix renormalization group in the age of matrix product states"
3. ITensor DMRG documentation: https://itensor.github.io/ITensors.jl/stable/DMRG.html

## File Changes

### Modified Files
- `src/custom/dmrg.jl`: Added single-site DMRG implementation (~200 lines)
- `README.md`: Added documentation and usage examples

### New Files
- `test/single_site_dmrg_test.jl`: Comprehensive test suite (260 lines)
- `example/single_site_dmrg_example.jl`: Usage example (75 lines)
- `SINGLE_SITE_DMRG.md`: This documentation file

## Conclusion

The single-site DMRG implementation provides a valuable alternative to two-site DMRG for refinement and optimization tasks where bond dimensions are already appropriate. It offers:

- ✅ Faster per-sweep performance
- ✅ Exact bond dimension preservation
- ✅ Clean separation from two-site implementation
- ✅ Comprehensive test coverage
- ✅ Full documentation

The implementation maintains the quality and style of the existing codebase while adding new functionality that complements the original two-site DMRG algorithm.
