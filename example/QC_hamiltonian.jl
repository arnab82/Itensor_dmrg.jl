using PyCall
using LinearAlgebra

pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")

# Create Simple Molecule
mol = pyscf.gto.M(; atom="""C      0.00000    0.00000    0.00000
  H      0.00000    0.00000    1.08900
  H      1.02672    0.00000   -0.36300
  H     -0.51336   -0.88916   -0.36300
  H     -0.51336    0.88916   -0.36300""", basis="sto3g", verbose=3)

# Run HF
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

# Create shorthands for 1- and 2-body integrals in MO basis
mo = mf.mo_coeff
n = size(mo, 1)
one_body = mo' * mf.get_hcore() * mo
two_body = reshape(mol.ao2mo(mf.mo_coeff; aosym=1), n, n, n, n)

# FCI (i.e. exact diagonalization)
cisolver = fci.FCI(mf)
cisolver.kernel()
println("FCI Energy (Ha): ", cisolver.e_tot)

# Setup for MPS Calculation
t = one_body
V = 0.5 * permutedims(two_body, (3, 2, 1, 4))
n_occ = mf.mo_occ
e_nuclear = mf.energy_nuc()

# Create MPO for the chemistry Hamiltonian
function chemistry_hamiltonian(n::Int, t::Matrix, V::Array{Float64,4}, e_nuclear::Float64)
    d = 4  # Local Hilbert space dimension (empty, up, down, up+down)
    
    # Initialize MPO tensors
    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, n)
    
    # Define operators
    I = Matrix{ComplexF64}(I, d, d)
    Cup = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Cdagup = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Cdn = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Cdagdn = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    
    # Construct MPO tensors
    for i in 1:n
        if i == 1
            mpo_tensors[i] = zeros(ComplexF64, d, d, 1, n^2 + 2n + 2)
            mpo_tensors[i][:,:,1,1] = I
            for j in 1:n
                mpo_tensors[i][:,:,1,j+1] = t[i,j] * Cdagup
                mpo_tensors[i][:,:,1,j+n+1] = t[i,j] * Cdagdn
            end
            for j in 1:n, k in 1:n
                idx = 2n + 1 + (j-1)*n + k
                mpo_tensors[i][:,:,1,idx] = V[i,j,k,i] * Cdagup * Cup + V[i,j,k,i] * Cdagdn * Cdn
            end
            mpo_tensors[i][:,:,1,end] = e_nuclear * I + sum(V[i,i,j,j] for j in 1:n) * (Cdagup*Cup + Cdagdn*Cdn)
        elseif i == n
            mpo_tensors[i] = zeros(ComplexF64, d, d, n^2 + 2n + 2, 1)
            mpo_tensors[i][:,:,1,1] = I
            for j in 1:n
                mpo_tensors[i][:,:,j+1,1] = Cup
                mpo_tensors[i][:,:,j+n+1,1] = Cdn
            end
            for j in 1:n, k in 1:n
                idx = 2n + 1 + (j-1)*n + k
                mpo_tensors[i][:,:,idx,1] = V[j,k,i,i] * Cdagup * Cup + V[j,k,i,i] * Cdagdn * Cdn
            end
            mpo_tensors[i][:,:,end,1] = I
        else
            mpo_tensors[i] = zeros(ComplexF64, d, d, n^2 + 2n + 2, n^2 + 2n + 2)
            mpo_tensors[i][:,:,1,1] = I
            for j in 1:n
                mpo_tensors[i][:,:,j+1,1] = Cup
                mpo_tensors[i][:,:,j+n+1,1] = Cdn
            end
            for j in 1:n, k in 1:n
                idx = 2n + 1 + (j-1)*n + k
                mpo_tensors[i][:,:,idx,1] = V[j,k,i,i] * Cdagup * Cup + V[j,k,i,i] * Cdagdn * Cdn
            end
            for j in 1:n
                mpo_tensors[i][:,:,end,j+1] = t[i,j] * Cdagup
                mpo_tensors[i][:,:,end,j+n+1] = t[i,j] * Cdagdn
            end
            for j in 1:n, k in 1:n
                idx = 2n + 1 + (j-1)*n + k
                mpo_tensors[i][:,:,end,idx] = V[i,j,k,i] * Cdagup * Cup + V[i,j,k,i] * Cdagdn * Cdn
            end
            mpo_tensors[i][:,:,end,end] = I
        end
    end
    
    return MPO(mpo_tensors)
end

H = chemistry_hamiltonian(n, t, V, e_nuclear)

# Initialize MPS
function random_MPS(n::Int, d::Int, χ::Int)
    tensors = [randn(ComplexF64, (i == 1 ? 1 : χ, d, i == n ? 1 : χ)) for i in 1:n]
    return MPS(tensors)
end

ψ0 = random_MPS(n, 4, 40)  # d = 4 for empty, up, down, up+down

# Run DMRG
function simple_dmrg(H::MPO, ψ0::MPS, max_sweeps::Int, χ_max::Int, cutoff::Float64)
    ψ = copy(ψ0)
    N = length(ψ.tensors)
    energy = 0.0
    
    for sweep in 1:max_sweeps
        # Right sweep
        for i in 1:(N-1)
            two_site_tensor = contract_two_sites(ψ, i)
            H_eff = construct_effective_hamiltonian(H, ψ, i)
            energy, ground_state = eigsolve(H_eff, vec(two_site_tensor), 1, :SR)
            energy = real(energy[1])
            ground_state = reshape(ground_state[1], size(two_site_tensor))
            
            U, S, V = svd(ground_state)
            χ = min(χ_max, count(S .> cutoff))
            U = U[:, 1:χ]
            S = S[1:χ]
            V = V[:, 1:χ]
            
            ψ.tensors[i] = reshape(U, (size(ψ.tensors[i], 1), 4, χ))
            ψ.tensors[i+1] = reshape(Diagonal(S) * V', (χ, 4, size(ψ.tensors[i+1], 3)))
        end
        
        # Left sweep
        for i in (N-1):-1:1
            two_site_tensor = contract_two_sites(ψ, i)
            H_eff = construct_effective_hamiltonian(H, ψ, i)
            energy, ground_state = eigsolve(H_eff, vec(two_site_tensor), 1, :SR)
            energy = real(energy[1])
            ground_state = reshape(ground_state[1], size(two_site_tensor))
            
            U, S, V = svd(ground_state)
            χ = min(χ_max, count(S .> cutoff))
            U = U[:, 1:χ]
            S = S[1:χ]
            V = V[:, 1:χ]
            
            ψ.tensors[i] = reshape(U * Diagonal(S), (size(ψ.tensors[i], 1), 4, χ))
            ψ.tensors[i+1] = reshape(V', (χ, 4, size(ψ.tensors[i+1], 3)))
        end
        
        println("Sweep $sweep, Energy = $energy")
    end
    
    return energy, ψ
end

# Run DMRG
max_sweeps = 10
χ_max = 300
cutoff = 1e-7
energy, ψ = simple_dmrg(H, ψ0, max_sweeps, χ_max, cutoff)

println("DMRG Energy (Ha): ", energy)
println("DMRG Error (Ha): ", abs(energy - cisolver.e_tot))