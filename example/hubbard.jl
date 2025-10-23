using ITensors, ITensorMPS
using Itensor_dmrg

# Define lattice parameters
Nx, Ny = 4, 4
N = Nx * Ny
t=1.0
U=4.0

sites = siteinds("Electron", N; conserve_qns=true)

# Create Hubbard model Hamiltonian (MPO)
H = Itensor_dmrg.hubbard_hamiltonian(sites, t, U)

# Define initial state (half-filling)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = productMPS(sites, state)

# DMRG calculation
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 100, 100, 200,200,200,200,200)
setcutoff!(sweeps, 1E-10)

energy, ψ = dmrg(H, ψ, sweeps)

# println("Ground state energy = ", energy)
# display(ψ)
println("H site indices: ", [siteinds(H, i) for i in 1:length(H)])
println("ψ site indices: ", [siteinds(ψ, i) for i in 1:length(ψ)])

energy, ψ = Itensor_dmrg.simple_dmrg(H, ψ, 2, maxdim=3, cutoff=1E-6)
println("Ground state energy = ", energy)

display(ψ)
