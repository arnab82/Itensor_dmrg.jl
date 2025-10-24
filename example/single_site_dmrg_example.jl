"""
Example: Single-Site DMRG for Heisenberg Model

This example demonstrates the use of single-site DMRG optimization
for a 1D Heisenberg spin chain.

Single-site DMRG is faster than two-site DMRG but preserves bond dimensions,
making it ideal for refinement after initial optimization.
"""

include("../src/custom/custom_dmrg.jl")

println("=" ^ 60)
println("Single-Site DMRG Example: Heisenberg Model")
println("=" ^ 60)

# System parameters
N = 10          # Number of sites
d = 2           # Physical dimension (spin-1/2)
chi_mpo = 5     # MPO bond dimension
chi_mps = 20    # MPS bond dimension (fixed for single-site DMRG)

println("\nSystem Parameters:")
println("  Number of sites (N): $N")
println("  Physical dimension (d): $d")
println("  MPS bond dimension (χ): $chi_mps")

# Create Heisenberg Hamiltonian
println("\nCreating Heisenberg Hamiltonian...")
H = heisenberg_ham(N, d, chi_mpo)

# Initialize random MPS
println("Initializing random MPS...")
mps = random_MPS(N, d, chi_mps)

# Record initial bond dimensions
initial_bonds = [size(mps.tensors[i], 3) for i in 1:N-1]
println("Initial bond dimensions: ", initial_bonds)

# DMRG parameters
max_sweeps = 50
tol = 1e-10

println("\nRunning single-site DMRG...")
println("  Max sweeps: $max_sweeps")
println("  Tolerance: $tol")
println()

# Run single-site DMRG
energy, optimized_mps = dmrg_single_site(H, mps, max_sweeps, tol)

# Verify bond dimensions are preserved
final_bonds = [size(optimized_mps.tensors[i], 3) for i in 1:N-1]

println("\n" * "=" ^ 60)
println("Results:")
println("=" ^ 60)
println("Ground state energy: ", energy)
println("Initial bond dimensions: ", initial_bonds)
println("Final bond dimensions:   ", final_bonds)
println("Bond dimensions preserved: ", initial_bonds == final_bonds)

# Compare with two-site DMRG for reference
println("\n" * "=" ^ 60)
println("Comparison with Two-Site DMRG:")
println("=" ^ 60)

mps_two_site = random_MPS(N, d, chi_mps)
max_sweeps_two = 20
chi_max = chi_mps

println("Running two-site DMRG for comparison...")
energy_two_site, _ = dmrg(H, mps_two_site, max_sweeps_two, chi_max, tol, false)

println("\nEnergy comparison:")
println("  Single-site DMRG: ", energy)
println("  Two-site DMRG:    ", energy_two_site)
println("  Energy difference: ", abs(energy - energy_two_site))

println("\n" * "=" ^ 60)
println("Single-site DMRG is faster per sweep but preserves bond dimensions.")
println("Use it for refinement after initial two-site DMRG optimization!")
println("=" ^ 60)
