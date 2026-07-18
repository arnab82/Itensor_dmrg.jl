"""
25-Site DMRG Demonstration

This script demonstrates that the custom DMRG code can successfully
handle 25-site calculations using two-site DMRG.

Usage:
    julia --project=. example/25_site_dmrg_demo.jl
"""

# Use absolute path relative to this file's location
example_dir = @__DIR__
project_dir = dirname(example_dir)
include(joinpath(project_dir, "src", "custom", "custom_dmrg.jl"))

println("=" ^ 70)
println("25-Site Custom DMRG Demonstration")
println("=" ^ 70)

# System parameters
N = 25          # Number of sites
d = 2           # Physical dimension (spin-1/2)
chi_mpo = 5     # MPO bond dimension
chi_mps = 20    # MPS bond dimension

println("\nSystem Parameters:")
println("  Number of sites: $N")
println("  Physical dimension: $d")
println("  MPS bond dimension: $chi_mps")

# Create Heisenberg Hamiltonian
println("\nCreating Heisenberg Hamiltonian...")
H = heisenberg_ham(N, d, chi_mpo)

# Initialize random MPS
println("Initializing random MPS...")
mps = random_MPS(N, d, chi_mps)

# DMRG parameters
max_sweeps = 10
chi_max = chi_mps
tol = 1e-8

println("\nRunning two-site DMRG...")
println("  Max sweeps: $max_sweeps")
println("  Bond dimension: $chi_max")
println("  Tolerance: $tol")
println()

# Run two-site DMRG
energy, mps_result = dmrg(H, mps, max_sweeps, chi_max, tol, false)

println("\n" * "=" ^ 70)
println("Results:")
println("=" ^ 70)
println("Ground state energy: ", energy)
println("\n25-site custom DMRG completed successfully!")
println("=" ^ 70)

println("\nNote: For 25-site calculations, two-site DMRG is recommended.")
println("Single-site DMRG currently has known issues and should be avoided.")
