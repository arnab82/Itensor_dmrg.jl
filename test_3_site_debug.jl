#!/usr/bin/env julia
"""
Debug script for 3-site Heisenberg chain
"""

using LinearAlgebra
using Printf

# Include custom DMRG implementation
include("./src/custom/custom_dmrg.jl")

function test_3_site()
    println("="^70)
    println("Testing 3-Site Heisenberg Chain")
    println("="^70)
    
    # System parameters
    N = 3  # 3 sites
    d = 2  # Physical dimension
    χ_mpo = 5  # MPO bond dimension
    
    println("\nSystem Parameters:")
    println("  Number of sites: $N")
    println("  Physical dimension: $d")
    
    # Create Heisenberg Hamiltonian
    println("\nCreating Heisenberg Hamiltonian...")
    H = heisenberg_ham(N, d, χ_mpo)
    
    # Exact Diagonalization
    println("\nExact Diagonalization:")
    H_matrix = MPO_to_array(H)
    println("  Matrix size: $(size(H_matrix))")
    
    hermitian_error = norm(H_matrix - H_matrix') / norm(H_matrix)
    println("  Hermiticity check: $hermitian_error")
    
    eigenvalues = eigvals(Hermitian((H_matrix + H_matrix') / 2))
    exact_energy = minimum(real(eigenvalues))
    
    println("  Ground state energy: $exact_energy")
    
    # DMRG with more verbose output
    println("\nDMRG Optimization:")
    χ_init = 8
    max_sweeps = 30
    χ_max = 20
    tol = 1e-10
    
    println("  Initial bond dimension: $χ_init")
    println("  Maximum sweeps: $max_sweeps")
    println("  Maximum bond dimension: $χ_max")
    
    mps = random_MPS(N, d, χ_init)
    dmrg_energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol, false)
    
    # Results
    println("\n" * "="^70)
    println("RESULTS:")
    @printf("  Exact energy: %.12f\n", exact_energy)
    @printf("  DMRG energy:  %.12f\n", dmrg_energy)
    @printf("  Error:        %.12f (%.2f%%)\n", abs(dmrg_energy - exact_energy), 
            100 * abs(dmrg_energy - exact_energy) / abs(exact_energy))
    println("="^70)
    
    return dmrg_energy, exact_energy
end

test_3_site()
