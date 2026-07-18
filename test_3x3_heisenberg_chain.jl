#!/usr/bin/env julia
"""
Test script for 3×3 Heisenberg chain (9 sites in 1D configuration)
This script runs custom DMRG and compares with exact diagonalization
"""

using LinearAlgebra
using Printf

# Include custom DMRG implementation
include("./src/custom/custom_dmrg.jl")

function test_3x3_heisenberg_chain()
    println("="^70)
    println("Testing 3×3 Heisenberg Chain (9-site 1D configuration)")
    println("="^70)
    
    # System parameters
    N = 9  # 3×3 = 9 sites in 1D chain
    d = 2  # Physical dimension (spin up, spin down)
    χ_mpo = 5  # MPO bond dimension
    
    println("\nSystem Parameters:")
    println("  Number of sites: $N")
    println("  Physical dimension: $d")
    println("  MPO bond dimension: $χ_mpo")
    
    # Create Heisenberg Hamiltonian
    println("\nCreating Heisenberg Hamiltonian...")
    H = heisenberg_ham(N, d, χ_mpo)
    
    # Exact Diagonalization
    println("\n" * "-"^70)
    println("EXACT DIAGONALIZATION")
    println("-"^70)
    
    println("Converting MPO to full matrix...")
    H_matrix = MPO_to_array(H)
    println("  Matrix size: $(size(H_matrix))")
    
    # Check Hermiticity
    hermitian_error = norm(H_matrix - H_matrix') / norm(H_matrix)
    println("  Hermiticity check: $hermitian_error")
    
    if hermitian_error > 1e-10
        @warn "Hamiltonian is not Hermitian! Relative error: $hermitian_error"
    end
    
    println("\nDiagonalizing...")
    eigenvalues = eigvals(Hermitian((H_matrix + H_matrix') / 2))
    exact_energy = minimum(real(eigenvalues))
    exact_energy_per_bond = exact_energy / (N-1)
    
    println("  Ground state energy: $exact_energy")
    println("  Energy per bond: $exact_energy_per_bond")
    
    # DMRG
    println("\n" * "-"^70)
    println("DMRG OPTIMIZATION")
    println("-"^70)
    
    # DMRG parameters
    χ_init = 10
    max_sweeps = 50
    χ_max = 50
    tol = 1e-8
    
    println("DMRG Parameters:")
    println("  Initial bond dimension: $χ_init")
    println("  Maximum sweeps: $max_sweeps")
    println("  Maximum bond dimension: $χ_max")
    println("  Convergence tolerance: $tol")
    
    println("\nInitializing random MPS...")
    mps = random_MPS(N, d, χ_init)
    
    println("Running DMRG...")
    dmrg_energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol, false)
    dmrg_energy_per_bond = dmrg_energy / (N-1)
    
    # Comparison
    println("\n" * "="^70)
    println("RESULTS COMPARISON")
    println("="^70)
    
    @printf("  Exact energy:        %.12f\n", exact_energy)
    @printf("  DMRG energy:         %.12f\n", dmrg_energy)
    @printf("  Absolute error:      %.12f\n", abs(dmrg_energy - exact_energy))
    @printf("  Relative error:      %.6f%%\n", 100 * abs(dmrg_energy - exact_energy) / abs(exact_energy))
    @printf("\n")
    @printf("  Exact energy/bond:   %.12f\n", exact_energy_per_bond)
    @printf("  DMRG energy/bond:    %.12f\n", dmrg_energy_per_bond)
    
    # Success criterion
    tolerance = 1e-3
    println("\n" * "="^70)
    if abs(dmrg_energy - exact_energy) < tolerance
        println("✓ TEST PASSED: DMRG energy matches exact energy within tolerance")
    else
        println("✗ TEST FAILED: DMRG energy does not match exact energy")
        println("  Note: There is a known issue with DMRG convergence for N>2 sites")
        println("  that requires further investigation.")
    end
    println("="^70)
    
    return dmrg_energy, exact_energy
end

# Run the test
test_3x3_heisenberg_chain()
