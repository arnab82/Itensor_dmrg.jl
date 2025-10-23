#!/usr/bin/env julia
"""
Test script for 2-site Heisenberg chain (should work correctly)
"""

using LinearAlgebra
using Printf

# Include custom DMRG implementation
include("./src/custom/custom_dmrg.jl")

function test_2_site()
    println("="^70)
    println("Testing 2-Site Heisenberg Chain")
    println("="^70)
    
    # System parameters
    N = 2
    d = 2
    χ_mpo = 5
    
    println("\nSystem Parameters:")
    println("  Number of sites: $N")
    
    # Create Heisenberg Hamiltonian
    println("\nCreating Heisenberg Hamiltonian...")
    H = heisenberg_ham(N, d, χ_mpo)
    
    # Exact Diagonalization
    println("\nExact Diagonalization:")
    H_matrix = MPO_to_array(H)
    eigenvalues = eigvals(Hermitian((H_matrix + H_matrix') / 2))
    exact_energy = minimum(real(eigenvalues))
    
    println("  Ground state energy: $exact_energy")
    
    # DMRG
    println("\nDMRG Optimization:")
    χ_init = 2
    max_sweeps = 10
    χ_max = 10
    tol = 1e-10
    
    mps = random_MPS(N, d, χ_init)
    dmrg_energy, _ = dmrg(H, mps, max_sweeps, χ_max, tol, false)
    
    # Results
    println("\n" * "="^70)
    println("RESULTS:")
    @printf("  Exact energy: %.12f\n", exact_energy)
    @printf("  DMRG energy:  %.12f\n", dmrg_energy)
    @printf("  Error:        %.12e (%.4f%%)\n", abs(dmrg_energy - exact_energy), 
            100 * abs(dmrg_energy - exact_energy) / abs(exact_energy))
    
    if abs(dmrg_energy - exact_energy) < 1e-6
        println("  ✓ TEST PASSED!")
    else
        println("  ✗ TEST FAILED!")
    end
    println("="^70)
    
    return dmrg_energy, exact_energy
end

test_2_site()
