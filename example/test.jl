using LinearAlgebra
using Random
include("./../src/custom/custom_dmrg.jl")
# Define dimensions
N = 20  # Number of sites
d1 = 4  # Physical dimension for MPO
d2 = 4  # Physical dimension for MPS
chi = 15  # Bond dimension

# Function to create a random MPS
function random_mps(N, d, χ)
    tensors = [rand(Complex{Float64}, χ, d, χ) for _ in 2:N-1]
    
    # Adjust the first and last tensors
    tensors = vcat(
        [rand(Complex{Float64}, 1, d, χ)],    # First tensor (χ_in = 1)
        tensors,
        [rand(Complex{Float64}, χ, d, 1)]     # Last tensor (χ_out = 1)
    )
    
    return MPS(tensors)
end

# Function to create a random MPO with shape checks
function random_mpo(N, d, χ)
    # Define and print the first tensor
    tensors = [rand(Complex{Float64}, 1, d, d, χ)]
    println("First MPO tensor shape: ", size(tensors[1]))  # Expected: (1, d, d, χ)
    
    # Intermediate tensors
    for _ in 2:N-1
        push!(tensors, rand(Complex{Float64}, χ, d, d, χ))
    end
    println("Intermediate MPO tensor shape: ", size(tensors[2]))  # Expected: (χ, d, d, χ)
    
    # Define and print the last tensor
    push!(tensors, rand(Complex{Float64}, χ, d, d, 1))
    println("Last MPO tensor shape: ", size(tensors[end]))  # Expected: (χ, d, d, 1)
    
    return MPO(tensors)
end



# Initialize random MPS and MPO
mps = random_mps(N, d1, chi)
H = random_mpo(N, d1, chi)
# println("MPS tensors: ", mps.tensors[1])
# println("MPO tensors: ", H.tensor[1])
# println(size(H.tensor[1]))
# println(size(mps.tensors[1]))



# L_dict, R_dict =  build_tensor_by_contracting(mps, H)
# H_eff=construct_effective_hamiltonian(H, mps, 3,chi)
# display(H_eff)
# max_sweeps = 100  # Maximum DMRG sweeps
# χ_max = 15   # Max bond dimension for truncation
# tol = 1e-12     # Tolerance for convergence
# energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol)
# Example parameters
Nx = 4         # Number of sites along the x-axis (length of chain)
Ny = 1         # For a 1D chain, set Ny = 1; for 2D grid, Ny > 1
t = 1.0        # Hopping parameter
U = 4.0        # On-site interaction
max_sweeps = 100  # Maximum DMRG sweeps
χ_max = 5   # Max bond dimension for truncation
tol = 1e-6     # Tolerance for convergence

# # Create Hubbard Hamiltonian as MPO
# H = hubbard(Nx=Nx, Ny=Ny, t=t, U=U, yperiodic=false)

# display(H)
# # Initialize a random MPS with bond dimension χ_max
# mps = random_mps(Nx * Ny, 4, χ_max)  # 4 corresponds to local dimension `d`

# # Run the DMRG algorithm
# energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol,hubbard)

# println("Ground state energy: ", energy)

# Define the chain length
# N = 10
# chi=5
# d=2
# # Generate the MPO for the Heisenberg Hamiltonian with 4 sites
# mpo = heisenberg_ham(N,d,chi)
# d1=2
# mps = random_mps(N, d1, chi)
# # Display each tensor shape in the MPO to verify dimensions
# println("MPO Tensors for Heisenberg Hamiltonian with N=$N:")
# for i in 1:N
#     println("Tensor at site $i: size ", size(mpo.tensor[i]))
# end
# energy, ground_state_mps = dmrg(mpo, mps, max_sweeps, χ_max, tol,hubbard)
