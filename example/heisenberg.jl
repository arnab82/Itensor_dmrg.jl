include("../src/Itensor/Itensor_dmrg.jl")
using ITensors
# Define lattice parameters
Nx, Ny = 4, 4
N = Nx * Ny

# Define model parameter
J = 1.234

# Create the Heisenberg Hamiltonian
H_heisenberg = Itensor_dmrg.heisenberg_hamiltonian(Nx, Ny, J)

# Define site indices for the lattice
s = siteinds("S=1/2", N)

# Convert OpSum to MPO
H = MPO(H_heisenberg, s)

# # Create an initial MPS (all spins up)
# ψ = productMPS(s, "Up")

# Create a random initial MPS in the Sz = 0 sector
N = length(s)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = randomMPS(s, state)
# DMRG parameters
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 100, 100, 200)
setcutoff!(sweeps, 1E-10)

# Run DMRG
energy, ψ = dmrg(H, ψ, sweeps)

println("Ground state energy = ", energy)

# Calculate magnetization
Sz_total = sum(expect(ψ, "Sz"))
println("Total Sz = ", Sz_total)
energy, ψ = Itensor_dmrg.simple_dmrg(H, ψ, 2, maxdim=2, cutoff=1E-6)
println("Ground state energy = ", energy)

display(ψ)