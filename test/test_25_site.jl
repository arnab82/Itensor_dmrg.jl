"""
    Test 25-site DMRG calculations

This test verifies that both two-site and single-site DMRG work for 25-site systems.
"""

using Test
using LinearAlgebra

# Include the custom DMRG implementation
include("../src/custom/custom_dmrg.jl")

@testset "25-Site DMRG Tests" begin
    
    @testset "25-site Heisenberg with Two-Site DMRG" begin
        N = 25
        d = 2
        chi_mps = 20
        chi_mpo = 5
        
        println("\nTesting 25-site Heisenberg model with two-site DMRG...")
        
        # Create Hamiltonian
        H = heisenberg_ham(N, d, chi_mpo)
        
        # Initialize MPS
        mps = random_MPS(N, d, chi_mps)
        
        # Run two-site DMRG with more sweeps
        max_sweeps = 30
        tol = 1e-6  # Relaxed tolerance
        
        energy, mps_result = dmrg(H, mps, max_sweeps, chi_mps, tol, false)
        
        # Test that energy is a real number
        @test energy isa Real
        @test isfinite(energy)
        
        # Test that ground state is returned
        @test mps_result isa MPS
        @test mps_result.N == N
        
        # Energy should be negative for Heisenberg model
        @test energy < 0
        
        # Energy should be reasonable (around -N*0.4 to -N*0.8 for 1D Heisenberg)
        @test energy > -N * 2  # Upper bound (very conservative)
        @test energy < N * 1   # Not positive
        
        println("  Final energy: $energy")
        println("  Energy per site: $(energy/N)")
    end
    
    @testset "25-site Heisenberg with Single-Site DMRG Refinement" begin
        N = 25
        d = 2
        chi_mps = 10  # Smaller for faster test
        chi_mpo = 5
        
        println("\nTesting 25-site with two-site DMRG followed by single-site refinement...")
        
        # Create Hamiltonian
        H = heisenberg_ham(N, d, chi_mpo)
        
        # Initialize MPS
        mps = random_MPS(N, d, chi_mps)
        
        # First, run two-site DMRG to establish bond dimensions
        max_sweeps_two = 10
        tol = 1e-6
        
        energy_two, mps_two = dmrg(H, mps, max_sweeps_two, chi_mps, tol, false)
        
        println("  After two-site DMRG: energy = $energy_two")
        
        # Then refine with single-site DMRG
        max_sweeps_single = 20
        energy_single, mps_single = dmrg_single_site(H, mps_two, max_sweeps_single, tol)
        
        println("  After single-site DMRG: energy = $energy_single")
        
        # Test that both energies are valid
        @test energy_two isa Real
        @test energy_single isa Real
        @test isfinite(energy_two)
        @test isfinite(energy_single)
        
        # Both should be negative for Heisenberg
        @test energy_two < 0
        @test energy_single < 0
        
        # Single-site DMRG should maintain reasonable energy (not collapse to zero)
        # Check it's within an order of magnitude
        @test abs(energy_single) > abs(energy_two) * 0.1
        @test abs(energy_single) < abs(energy_two) * 10
        
        # Test that ground states are returned
        @test mps_two isa MPS
        @test mps_single isa MPS
        @test mps_single.N == N
        
        # Verify bond dimensions are preserved in single-site DMRG
        bond_dims_two = [size(mps_two.tensors[i], 3) for i in 1:N-1]
        bond_dims_single = [size(mps_single.tensors[i], 3) for i in 1:N-1]
        @test bond_dims_two == bond_dims_single
    end
    
    @testset "Smaller System Convergence Test" begin
        N = 10  # Smaller system for testing convergence
        d = 2
        chi = 15
        
        println("\nTesting 10-site system for better convergence...")
        
        H = heisenberg_ham(N, d, 5)
        mps = random_MPS(N, d, chi)
        
        # Run DMRG with enough sweeps to converge
        max_sweeps = 30
        tol = 1e-8
        
        energy, mps_result = dmrg(H, mps, max_sweeps, chi, tol, false)
        
        # Test that energy is reasonable
        @test energy isa Real
        @test isfinite(energy)
        @test energy < 0
        
        # For 10-site 1D Heisenberg, energy should be around -4 to -5
        @test energy > -N * 1
        @test energy < 0
        
        println("  Final energy: $energy")
        println("  Energy per site: $(energy/N)")
    end
end
