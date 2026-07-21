using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

# Import (not `using`) so the module name is available for qualified access
# without pulling NaiveDMRG.MPS/MPO into scope, where they would collide with
# the ITensor `MPS`/`MPO` types the reference tests rely on.
import NaiveDMRG

@info "NaiveDMRG.jl test suite" julia = string(VERSION)

# `verbose = true` prints the per-testset result tree, so the log shows each
# step (environments, two-site DMRG, single-site DMRG, builders, observables,
# entanglement, scalar types, reference) rather than a single pass/fail count.
@testset verbose = true "NaiveDMRG.jl" begin
    @info "Contraction primitives vs dense references"
    include("naive_primitives_test.jl")

    @info "Two-site DMRG (exact diagonalization + ITensor agreement)"
    include("naive_dmrg_test.jl")

    @info "Single-site DMRG with subspace expansion"
    include("single_site_dmrg_test.jl")

    @info "Generic nearest-neighbor MPO builder"
    include("mpo_builder_test.jl")

    @info "1D Fermi-Hubbard chain (d=4) vs ITensor Electron reference"
    include("hubbard_test.jl")

    @info "Local observables and correlations"
    include("observables_test.jl")

    @info "Entanglement entropy and Schmidt spectrum"
    include("entanglement_test.jl")

    @info "Parametric scalar types (real Float64 pipeline)"
    include("parametric_types_test.jl")

    @info "ITensor reference baseline"
    include("reference_test.jl")
end
