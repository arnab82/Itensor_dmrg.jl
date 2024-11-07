using ITensors, ITensorMPS
# using Itensor_dmrg

include("./../src/hubbard_hamiltonian.jl")

# Define site indices for the lattice with fermion conservation
s = siteinds("Electron", N; conserve_qns=true)


# Create the Hamiltonian MPO
H = hubbard_hamiltonian(s, t, U)

# Define initial state (half-filling)
state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
ψ = productMPS(s, state)

# DMRG calculation
sweeps = Sweeps(50)
setmaxdim!(sweeps, 10, 20, 100, 100, 200,200,200,200,200)
setcutoff!(sweeps, 1E-10)

energy, ψ = dmrg(H, ψ, sweeps)

println("Ground state energy = ", energy)
display(ψ)
