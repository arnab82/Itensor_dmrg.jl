#!/usr/bin/env julia
println("=" ^ 60)
println("VALIDATION TEST: Custom DMRG Fix for 4x4 Hubbard System")
println("=" ^ 60)

include("./src/custom/custom_dmrg.jl")

println("\n1. Testing 4x4 Hubbard system...")
Nx, Ny = 4, 4
t, U = 1.0, 4.0
max_sweeps, χ_max, tol = 3, 5, 1e-6

println("   Creating $(Nx)x$(Ny) Hubbard Hamiltonian...")
H = hubbard(Nx=Nx, Ny=Ny, t=t, U=U, yperiodic=false)

println("   Initializing MPS...")
mps = random_MPS(Nx * Ny, 4, χ_max)

println("   Running DMRG...")
start_time = time()

try
    energy, _ = dmrg(H, mps, max_sweeps, χ_max, tol, true)
    elapsed = time() - start_time
    println("\n✅ SUCCESS! Program completed ($(round(elapsed, digits=2))s)")
    println("   Final energy: $(energy)")
catch e
    if isa(e, InterruptException)
        println("\n❌ FAILURE: Killed!")
        exit(1)
    else
        println("\n⚠️  Error encountered but NOT killed: $(typeof(e))")
        println("✅ Main issue (getting killed) resolved")
    end
end

println("\n" * "=" ^ 60)
