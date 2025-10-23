#!/usr/bin/env julia
# Validation script for 4x4 Heisenberg model with memory-safe parameters

using ITensors, ITensorMPS
using Itensor_dmrg

println("="^60)
println("Validating 4x4 Heisenberg Model with Safe Parameters")
println("="^60)

# Define lattice parameters
Nx, Ny = 4, 4
N = Nx * Ny
println("\nLattice size: $(Nx)x$(Ny) = $N sites")

# Define model parameter
J = 1.234
println("Coupling constant J = $J")

# Create the Heisenberg Hamiltonian
println("\nCreating Heisenberg Hamiltonian...")
H_heisenberg = Itensor_dmrg.heisenberg_hamiltonian(Nx, Ny, J)

# Define site indices for the lattice
s = siteinds("S=1/2", N)

# Convert OpSum to MPO
println("Converting to MPO...")
H = MPO(H_heisenberg, s)
println("MPO created with $(length(H)) sites")

# Create initial MPS in the Sz = 0 sector
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = productMPS(s, state)
println("Initial MPS created (alternating Up/Dn pattern)")

# DMRG parameters - conservative for 4x4 lattice
println("\nDMRG Parameters:")
sweeps = Sweeps(5)  # Reduced from 10 for faster validation
setmaxdim!(sweeps, 10, 20, 50, 100, 100)
setcutoff!(sweeps, 1E-10)
println("  Number of sweeps: 5")
println("  Max bond dimensions: [10, 20, 50, 100, 100]")
println("  Cutoff: 1E-10")

# Estimate memory requirements
d = 2  # Physical dimension for spin-1/2
χ_max = 100  # Maximum bond dimension
estimated_mem_mb = N * χ_max^2 * d^2 / 1e6
println("\nEstimated peak memory usage: ~$(round(estimated_mem_mb, digits=1)) MB")

# Run DMRG
println("\nStarting DMRG calculation...")
println("-"^60)

try
    energy, ψ = dmrg(H, ψ, sweeps)
    
    println("-"^60)
    println("\n✓ DMRG completed successfully!")
    println("Ground state energy = $energy")
    
    # Calculate magnetization
    Sz_total = sum(expect(ψ, "Sz"))
    println("Total Sz = $Sz_total")
    
    # Check bond dimensions
    max_bond_dim = maximum([dim(linkind(ψ, b)) for b in 1:(N-1)])
    println("Maximum bond dimension reached: $max_bond_dim")
    
    println("\n" * "="^60)
    println("VALIDATION PASSED: 4x4 Heisenberg model runs successfully")
    println("="^60)
    
catch e
    println("\n" * "="^60)
    println("ERROR: DMRG calculation failed")
    println("="^60)
    println("Error message: $e")
    rethrow(e)
end
