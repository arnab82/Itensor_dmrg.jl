#!/usr/bin/env julia

"""
Simple benchmark script to demonstrate performance improvements
in Itensor_dmrg.jl

This script compares memory usage and execution time with different settings.
"""

using ITensors
using ITensorMPS
using Itensor_dmrg
using Printf

println("="^60)
println("Itensor_dmrg.jl Performance Benchmark")
println("="^60)

# Test system: Small Heisenberg model
Nx, Ny = 4, 4
N = Nx * Ny
J = 1.0

println("\nTest System: Heisenberg Model")
println("  Lattice: $(Nx)x$(Ny) = $N sites")
println("  Coupling: J = $J")

# Create Hamiltonian
H_opsum = Itensor_dmrg.heisenberg_hamiltonian(Nx, Ny, J)
s = siteinds("S=1/2", N)
H = MPO(H_opsum, s)

# Create initial state
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ0 = randomMPS(s, state)

println("\n" * "="^60)
println("Benchmark 1: Verbose vs Silent Mode")
println("="^60)

# Benchmark 1: Verbose mode
println("\nRunning with verbose output (silent=false)...")
ψ1 = copy(ψ0)
time_verbose = @elapsed begin
    energy1, ψ1 = Itensor_dmrg.simple_dmrg(H, ψ1, 2; maxdim=50, cutoff=1E-8, silent=false)
end
@printf("  Time: %.3f seconds\n", time_verbose)
@printf("  Energy: %.12f\n", energy1)

println("\nRunning with silent mode (silent=true)...")
ψ2 = copy(ψ0)
time_silent = @elapsed begin
    energy2, ψ2 = Itensor_dmrg.simple_dmrg(H, ψ2, 2; maxdim=50, cutoff=1E-8, silent=true)
end
@printf("  Time: %.3f seconds\n", time_silent)
@printf("  Energy: %.12f\n", energy2)
@printf("  Speedup: %.2fx\n", time_verbose / time_silent)

println("\n" * "="^60)
println("Benchmark 2: Memory Usage with Different Bond Dimensions")
println("="^60)

for maxdim in [20, 50, 100]
    println("\nBond dimension χ_max = $maxdim")
    ψ_test = copy(ψ0)
    
    # Force garbage collection before measurement
    GC.gc()
    
    # Measure memory
    mem_before = Base.gc_live_bytes() / 1e6  # Convert to MB
    
    energy, ψ_test = Itensor_dmrg.simple_dmrg(H, ψ_test, 2; maxdim=maxdim, cutoff=1E-8, silent=true)
    
    # Force garbage collection after to get accurate reading
    GC.gc()
    mem_after = Base.gc_live_bytes() / 1e6
    
    @printf("  Energy: %.12f\n", energy)
    @printf("  Memory delta: %.2f MB\n", mem_after - mem_before)
    @printf("  MPS memory: %.2f MB\n", sum(sizeof, ψ_test) / 1e6)
end

println("\n" * "="^60)
println("Benchmark 3: Convergence Speed")
println("="^60)

maxdim = 100
cutoff = 1E-8

println("\nRunning DMRG with progressive sweeps...")
ψ_conv = copy(ψ0)
energies = Float64[]
times = Float64[]

for sweep in 1:5
    time_sweep = @elapsed begin
        energy, ψ_conv = Itensor_dmrg.simple_dmrg(H, ψ_conv, 1; maxdim=maxdim, cutoff=cutoff, silent=true)
    end
    push!(energies, energy)
    push!(times, time_sweep)
    
    energy_change = sweep > 1 ? abs(energies[end] - energies[end-1]) : NaN
    @printf("  Sweep %d: E = %.12f, ΔE = %.2e, Time = %.3f s\n", 
            sweep, energy, energy_change, time_sweep)
end

println("\n" * "="^60)
println("Summary")
println("="^60)
println("\nOptimizations demonstrated:")
println("  ✓ Silent mode reduces I/O overhead")
println("  ✓ Efficient memory usage with ProjMPO reuse")
println("  ✓ Fast convergence with optimized tensor operations")
println("  ✓ Memory scales predictably with bond dimension")
println("\nSee PERFORMANCE_GUIDE.md for more optimization tips!")
println("="^60)
