using ITensors, ITensorMPS
using Itensor_dmrg

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

# Create a random initial MPS in the Sz = 0 sector
N = length(s)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = productMPS(s, state)
# DMRG parameters
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 50, 100, 100)
setcutoff!(sweeps, 1E-10)

# Run DMRG
energy, ψ = dmrg(H, ψ, sweeps)

println("Ground state energy = ", energy)

# Calculate magnetization
Sz_total = sum(expect(ψ, "Sz"))
println("Total Sz = ", Sz_total)

# Optionally run custom simple_dmrg for comparison (using reasonable bond dimension)
# energy2, ψ2 = Itensor_dmrg.simple_dmrg(H, ψ, 2, maxdim=50, cutoff=1E-8)
# println("Ground state energy (simple_dmrg) = ", energy2)