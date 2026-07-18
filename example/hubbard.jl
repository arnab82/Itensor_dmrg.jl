using ITensors, ITensorMPS
using Itensor_dmrg

# Define lattice parameters
Nx, Ny = 4, 4
N = Nx * Ny
t=1.0
U=4.0

sites = siteinds("Electron", N; conserve_qns=true)

# Create Hubbard model Hamiltonian (MPO)
H = Itensor_dmrg.hubbard_hamiltonian(sites, t, U, Nx, Ny)

# Define initial state (half-filling)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = productMPS(sites, state)

# DMRG calculation
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 50, 100, 100)
setcutoff!(sweeps, 1E-10)

energy, ψ = dmrg(H, ψ, sweeps)

println("Ground state energy = ", energy)

# Optionally run custom simple_dmrg for comparison (using reasonable bond dimension)
# energy2, ψ2 = Itensor_dmrg.simple_dmrg(H, ψ, 2, maxdim=50, cutoff=1E-8)
# println("Ground state energy (simple_dmrg) = ", energy2)
