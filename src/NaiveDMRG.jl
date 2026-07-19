"""
    NaiveDMRG

A small, self-contained ("naive") finite-system, two-site DMRG for
open-boundary matrix-product states. The solver is written from scratch and
does not call ITensor: dense `ComplexF64` MPS/MPO tensors, matrix-free two-site
effective Hamiltonians, KrylovKit local eigensolves, and SVD bond truncation.

The from-scratch solver is the primary API of this package and is exported
directly, so a typical session is:

```julia
using NaiveDMRG
H = heisenberg_mpo(20)
psi0 = random_MPS(20, 2, 16)
energy, psi = dmrg(H, psi0; nsweeps=20, maxdim=64, cutoff=1e-10, tol=1e-8)
```

ITensor is retained only as an independent reference for regression tests and
numerical comparison. That code lives in the clearly-separated submodule
`NaiveDMRG.Reference` and is never used by the functions above.
"""
module NaiveDMRG

using LinearAlgebra
using Random
using KrylovKit
using TensorOperations
using Printf

# --- From-scratch ("naive") DMRG: the primary public API -------------------
include("core/MPS.jl")
include("core/MPO.jl")
include("core/heisenberg_ham.jl")
include("core/dmrg.jl")

export MPS, MPO, random_MPS, heisenberg_mpo, heisenberg_ham
export dmrg, dmrg!, compute_energy, overlap, dense, bond_dimensions
export DMRGResult, SweepRecord, LocalSolveRecord
export left_canonicalize!, right_canonicalize!, left_normalize!, right_normalize!

# --- ITensor reference implementation (regression comparisons only) --------
include("reference/Reference.jl")

end
