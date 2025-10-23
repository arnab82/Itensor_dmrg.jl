using LinearAlgebra
using Random
include("./../src/custom/custom_dmrg.jl")

# Example parameters for 2x2 Hubbard model
Nx = 2         # Number of sites along the x-axis
Ny = 2         # Number of sites along the y-axis  
N = Nx * Ny    # Total sites = 4
d = 4          # Physical dimension (empty, up, down, up+down)
t = 1.0        # Hopping parameter
U = 4.0        # On-site interaction

# DMRG parameters
max_sweeps = 10  # Maximum DMRG sweeps
χ_init = 4       # Initial bond dimension
χ_max = 10       # Max bond dimension for truncation
tol = 1e-8       # Tolerance for convergence

# Create Hubbard Hamiltonian as MPO
H = hubbard(Nx=Nx, Ny=Ny, t=t, U=U, yperiodic=false)

# Initialize a random MPS with bond dimension χ_init
mps = random_MPS(N, d, χ_init)

# Run the DMRG algorithm
energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol, true)

println("Ground state energy: ", energy)
println("Final MPS bond dimensions: ", [size(ground_state_mps.tensors[i], 3) for i in 1:N-1])

