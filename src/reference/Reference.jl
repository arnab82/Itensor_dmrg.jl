"""
    NaiveDMRG.Reference

ITensor-based Hamiltonians and DMRG kept *only* as an independent reference for
regression tests and numerical comparison against the from-scratch solver in
the parent [`NaiveDMRG`](@ref) module. Nothing in the primary `NaiveDMRG` API
calls into this submodule.

Because ITensor exports its own `MPS`/`MPO` types, this code is deliberately
isolated in its own namespace so it never collides with `NaiveDMRG.MPS` and
`NaiveDMRG.MPO`.
"""
module Reference

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using KrylovKit
using Combinatorics

include("hubbard_hamiltonian.jl")
include("heisenberg_hamiltonian.jl")
include("dmrg_ITensor.jl")
include("utils.jl")

export hubbard_hamiltonian
export heisenberg_hamiltonian
export compute_energy
export simple_dmrg
export svd_truncate

end
