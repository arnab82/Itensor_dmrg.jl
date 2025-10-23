using LinearAlgebra
using KrylovKit
using Einsum
using TensorOperations
using Printf

# Structure to cache left and right environments
mutable struct EnvironmentCache
    L::Dict{Int, Array{Complex{Float64}, 3}}
    R::Dict{Int, Array{Complex{Float64}, 3}}
    
    function EnvironmentCache()
        new(Dict{Int, Array{Complex{Float64}, 3}}(), Dict{Int, Array{Complex{Float64}, 3}}())
    end
end

# Initialize or update the environment cache
function initialize_cache!(cache::EnvironmentCache, H::MPO, mps::MPS)
    N = mps.N
    
    # Initialize left environment
    cache.L[0] = ones(Complex{Float64}, 1, 1, 1)
    
    # Build all left environments
    for i in 1:N-1
        mps_i = mps.tensors[i]
        mps_i_conj = conj(mps_i)
        H_i = H.tensor[i]
        L = cache.L[i-1]
        
        L_temp = zeros(Complex{Float64}, size(mps_i, 3), size(H_i, 4), size(mps_i, 3))
        @einsum L_temp[a, b, c] := L[χ1, χ2, χ3] * mps_i[χ1, d1, a] * H_i[χ2, d1, d2, b] * mps_i_conj[χ3, d2, c]
        cache.L[i] = L_temp
    end
    
    # Initialize right environment
    cache.R[N] = ones(Complex{Float64}, 1, 1, 1)
    
    # Build all right environments
    for i in N:-1:2
        mps_i = mps.tensors[i]
        mps_i_conj = conj(mps_i)
        H_i = H.tensor[i]
        R = cache.R[i]
        
        R_temp = zeros(Complex{Float64}, size(mps_i, 1), size(H_i, 1), size(mps_i, 1))
        @einsum R_temp[a, b, c] := R[χ1, χ2, χ3] * mps_i[a, d1, χ1] * H_i[b, d1, d2, χ2] * mps_i_conj[c, d2, χ3]
        cache.R[i-1] = R_temp
    end
end

# Update a single left environment entry after tensor update
function update_left_environment!(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)
    if i == 0
        cache.L[0] = ones(Complex{Float64}, 1, 1, 1)
        return
    end
    
    mps_i = mps.tensors[i]
    mps_i_conj = conj(mps_i)
    H_i = H.tensor[i]
    L = cache.L[i-1]
    
    # Preallocate output if needed, or reuse existing array
    out_size = (size(mps_i, 3), size(H_i, 4), size(mps_i, 3))
    if haskey(cache.L, i) && size(cache.L[i]) == out_size
        fill!(cache.L[i], zero(Complex{Float64}))
        L_temp = cache.L[i]
    else
        L_temp = zeros(Complex{Float64}, out_size)
    end
    
    @einsum L_temp[a, b, c] = L[χ1, χ2, χ3] * mps_i[χ1, d1, a] * H_i[χ2, d1, d2, b] * mps_i_conj[χ3, d2, c]
    cache.L[i] = L_temp
end

# Update a single right environment entry after tensor update
function update_right_environment!(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)
    N = mps.N
    if i == N
        cache.R[N] = ones(Complex{Float64}, 1, 1, 1)
        return
    end
    
    mps_i = mps.tensors[i]
    mps_i_conj = conj(mps_i)
    H_i = H.tensor[i]
    R = cache.R[i]
    
    # Preallocate output if needed, or reuse existing array
    out_size = (size(mps_i, 1), size(H_i, 1), size(mps_i, 1))
    if haskey(cache.R, i-1) && size(cache.R[i-1]) == out_size
        fill!(cache.R[i-1], zero(Complex{Float64}))
        R_temp = cache.R[i-1]
    else
        R_temp = zeros(Complex{Float64}, out_size)
    end
    
    @einsum R_temp[a, b, c] = R[χ1, χ2, χ3] * mps_i[a, d1, χ1] * H_i[b, d1, d2, χ2] * mps_i_conj[c, d2, χ3]
    cache.R[i-1] = R_temp
end

# Function to contract two adjacent sites in an MPS (right-moving)
function contract_two_sites_right(mps::MPS, i::Int)
    left_tensor = mps.tensors[i]
    right_tensor = mps.tensors[i+1]
    
    # Direct tensor contraction without intermediate allocation
    chi_left = size(left_tensor, 1)
    chi_right = size(right_tensor, 3)
    two_site_tensor = zeros(Complex{Float64}, chi_left, mps.d, mps.d, chi_right)
    
    @tensor two_site_tensor[l, s1, s2, r] = left_tensor[l, s1, k] * right_tensor[k, s2, r]
    
    # Reshape into a matrix
    return reshape(two_site_tensor, (chi_left * mps.d, mps.d * chi_right))
end

# Function to contract two adjacent sites in an MPS (left-moving)
function contract_two_sites_left(mps::MPS, i::Int)
    left_tensor = mps.tensors[i-1]
    right_tensor = mps.tensors[i]
    
    # Direct tensor contraction without intermediate allocation
    chi_left = size(left_tensor, 1)
    chi_right = size(right_tensor, 3)
    two_site_tensor = zeros(Complex{Float64}, chi_left, mps.d, mps.d, chi_right)
    
    @tensor two_site_tensor[l, s1, s2, r] = left_tensor[l, s1, k] * right_tensor[k, s2, r]
    
    # Reshape into a matrix
    return reshape(two_site_tensor, (chi_left * mps.d, mps.d * chi_right))
end


function build_tensor_by_contracting(mps::MPS, H::MPO)
    L = ones(Complex{Float64}, 1, 1, 1)
    R = ones(Complex{Float64}, 1, 1, 1)
    L_dict = Dict{Int, Array{Complex{Float64}, 3}}()
    R_dict = Dict{Int, Array{Complex{Float64}, 3}}()
    N=mps.N
    for i in 1:N
        # println(i)
        # println(size(L))
        # println(size(H.tensor[i]))
        # println(size(mps.tensors[i]))
        mps_i=mps.tensors[i]
        H_i=H.tensor[i]
        L_temp = zeros(Complex{Float64}, size(mps_i, 1), size(H_i, 2), size(mps_i, 1))
        @einsum L_temp[a, b, c] := L[ χ1,χ2 , χ3] * mps_i[χ1,d,a]* H_i[χ2, d, d, b] * mps_i[χ3, d, c]
        L = L_temp
        L_dict[i] = L_temp
    end
    for i in N:-1:1
        # println(i)
        # println(size(R))
        # println(size(H.tensor[i]))
        # println(size(mps.tensors[i]))
        mps_i = mps.tensors[i]
        H_i = H.tensor[i]
        R_temp = zeros(Complex{Float64}, size(mps_i, 1), size(H_i, 2), size(mps_i, 1))
        @einsum R_temp[a, b, c] := R[χ1, χ2, χ3] * mps_i[a, d,χ1 ] * H_i[b, d, d, χ2] * mps_i[c, d, χ3]
        R = R_temp
        R_dict[i] = R_temp
    end
    return L_dict, R_dict
end




"""

    construct_effective_hamiltonian(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)

Construct the effective Hamiltonian for a two-site DMRG optimization using cached environments.

This function builds the effective Hamiltonian for sites i and i+1 in a Matrix Product State (MPS) 
with respect to a Matrix Product Operator (MPO) Hamiltonian, using pre-computed left and right environments.

# Arguments
- `cache::EnvironmentCache`: Cache containing pre-computed environments
- `H::MPO`: The Hamiltonian in MPO form
- `mps::MPS`: The current Matrix Product State
- `i::Int`: The index of the first site of the two-site block

# Returns
- `H_eff::Matrix{Complex{Float64}}`: The effective Hamiltonian as a matrix

Construct the effective Hamiltonian for two adjacent sites in an MPS.
"""

function construct_effective_hamiltonian(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)
    # Get left and right environments from cache
    L = cache.L[i-1]
    R = cache.R[i+1]

    d = size(mps.tensors[i], 2)
    H_i = H.tensor[i]
    H_iplus1 = H.tensor[i+1]

    # Perform tensor contraction to build the effective Hamiltonian
    # Contract H_i and H_i+1
    H_two = zeros(Complex{Float64}, size(H_i, 1), size(H_i, 2), size(H_iplus1, 2), size(H_i, 3), size(H_iplus1, 3), size(H_iplus1, 4))
    @einsum H_two[a, b, c, d, e, f] = H_i[a, b, d, g] * H_iplus1[g, c, e, f]

    # Contract with left environment
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] = L[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] = temp1[a, b, c, d, e, f, i] * R[g, i, h]

    # Reshape temp2 to construct the effective Hamiltonian H_eff
    chi_L = size(temp2, 1)  # mps_left bond
    chi_R = size(temp2, 7)  # mps_right bond
    # Permute and reshape in one step for efficiency
    temp2_perm = permutedims(temp2, [2, 3, 4, 8, 1, 5, 6, 7])
    H_eff = reshape(temp2_perm, (chi_L * d * d * chi_R, chi_L * d * d * chi_R))

    return H_eff
end

function construct_effective_hamiltonian_left(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)
    # Get left and right environments from cache
    L = cache.L[i-2]
    R = cache.R[i]

    # Construct the effective Hamiltonian by contracting L, H_(i-1), H_i, and R
    d = size(mps.tensors[i], 2)
    H_im1 = H.tensor[i - 1]
    H_i = H.tensor[i]

    # Contract H_(i-1) and H_i
    H_two = zeros(Complex{Float64}, size(H_im1, 1), size(H_im1, 2), size(H_i, 2), size(H_im1, 3), size(H_i, 3), size(H_i, 4))
    @einsum H_two[a, b, c, d, e, f] = H_im1[a, b, d, g] * H_i[g, c, e, f]
    
    # Contract with left environment
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] = L[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] = temp1[a, b, c, d, e, f, i] * R[g, i, h]
    
    # Reshape temp2 to construct the effective Hamiltonian H_eff
    chi_L = size(temp2, 1)  # mps_left bond
    chi_R = size(temp2, 7)  # mps_right bond
    # Permute and reshape in one step for efficiency
    temp2_perm = permutedims(temp2, [2, 3, 4, 8, 1, 5, 6, 7])
    H_eff = reshape(temp2_perm, (chi_L * d * d * chi_R, chi_L * d * d * chi_R))

    return H_eff
end




# Function to solve the eigenvalue problem
function eigsolve(H_eff::Matrix{ComplexF64}, init_state::Vector{ComplexF64}, num_eigvals::Int, which_eigvals::Symbol)
    # Use KrylovKit's eigsolve function
    vals, vecs, info = KrylovKit.eigsolve(H_eff, init_state, num_eigvals, which_eigvals)
    return vals[1], vecs[1]  # Return the ground state energy and wavefunction
end


function dmrg_sweep!(H::MPO, mps::MPS, cache::EnvironmentCache, direction::Symbol, χ_max::Int, tol::Float64, hubbard=false)
    energy = 0.0
    if hubbard==true
        range = direction == :right ? (1:mps.N-1) : reverse(2:mps.N)
    else
        range = direction == :right ? (2:mps.N-2) : reverse(2:mps.N-2)
    end
    total_truncation_error = 0.0
    
    for i in range
        # Save tensor dimensions before any modifications
        if direction == :right
            chi_i_left = size(mps.tensors[i], 1)
            chi_iplus1_right = size(mps.tensors[i+1], 3)
        else
            chi_iminus1_left = size(mps.tensors[i-1], 1)
            chi_i_right = size(mps.tensors[i], 3)
        end
        
        # Contract two sites
        # Construct effective Hamiltonian using cached environments
        if direction == :right
            two_site_tensor = contract_two_sites_right(mps, i)
            H_eff = construct_effective_hamiltonian(cache, H, mps, i)
        else
            two_site_tensor = contract_two_sites_left(mps, i)
            H_eff = construct_effective_hamiltonian_left(cache, H, mps, i)
        end
        
        # Solve for ground state using eigsolve
        energy, ground_state = eigsolve(H_eff, vec(two_site_tensor), 1, :SR)

        # Reshape ground state back to a tensor (same shape as two_site_tensor)
        ground_state = reshape(ground_state, size(two_site_tensor))
        
        # Normalize the ground state (in-place where possible)
        ground_state_norm = norm(ground_state)
        ground_state ./= ground_state_norm
        
        # Perform SVD and truncate
        F = svd(ground_state)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncate bond dimension: keep at most χ_max singular values above tolerance
        χ_trunc = min(χ_max, length(S), count(>(tol), S))
        # Ensure at least one singular value is kept
        χ_trunc = max(1, χ_trunc)
        
        # Calculate truncation error before truncation
        @inbounds truncation_error = sum(abs2(S[i]) for i in (χ_trunc+1):length(S))
        total_truncation_error += truncation_error
        
        # Truncate arrays
        U = @view U[:, 1:χ_trunc]
        S_trunc = @view S[1:χ_trunc]
        Vt = @view Vt[1:χ_trunc, :]
        
        # Update MPS tensors
        if direction == :right
            mps.tensors[i] = reshape(U, (chi_i_left, mps.d, χ_trunc))
            mps.tensors[i+1] = reshape(Diagonal(S_trunc) * Vt, (χ_trunc, mps.d, chi_iplus1_right))
            # Update cached environments after updating tensors
            update_left_environment!(cache, H, mps, i)
        else
            mps.tensors[i-1] = reshape(U, (chi_iminus1_left, mps.d, χ_trunc))
            mps.tensors[i] = reshape(Diagonal(S_trunc) * Vt, (χ_trunc, mps.d, chi_i_right))
            # Update cached environments after updating tensors
            update_right_environment!(cache, H, mps, i)
        end
    end
    
    return real(energy), total_truncation_error, mps
end


function dmrg(H::MPO, mps::MPS, max_sweeps::Int, χ_max::Int, tol::Float64, hubbard=false)
    energy = 0.0
    prev_energy = 0.0
    
    # Initialize environment cache
    cache = EnvironmentCache()
    initialize_cache!(cache, H, mps)
    
    for sweep in 1:max_sweeps
        # Right sweep
        energy_right, trunc_error_right, mps = dmrg_sweep!(H, mps, cache, :right, χ_max, tol, hubbard)
        
        # Only reinitialize cache if truncation was significant or first sweep
        # This saves computation while maintaining accuracy
        if sweep == 1 || trunc_error_right > tol * 10
            initialize_cache!(cache, H, mps)
        end
        
        # Left sweep
        energy_left, trunc_error_left, mps = dmrg_sweep!(H, mps, cache, :left, χ_max, tol, hubbard)
        
        # Only reinitialize if needed (not on last sweep and if truncation significant)
        if sweep < max_sweeps && trunc_error_left > tol * 10
            initialize_cache!(cache, H, mps)
        end
        
        # Total truncation error for this sweep
        total_truncation_error = trunc_error_right + trunc_error_left
        
        # Check for convergence based on energy change
        energy_change = abs(energy_left - prev_energy)
        avg_sweep_energy = (energy_right + energy_left) / 2
        
        @printf("Sweep %d completed. Energy = %.12f, Energy Change = %.12e, Truncation Error = %.12e\n", 
                sweep, avg_sweep_energy, energy_change, total_truncation_error)
        
        # Check convergence
        if sweep > 1 && energy_change < tol
            @printf("Converged after %d sweeps. Final Energy = %.12f, Final Truncation Error = %.12e\n", 
                    sweep, energy_left, total_truncation_error)
            return energy_left, mps
        end
        
        # Update energy
        prev_energy = energy_left
    end
    
    @warn "DMRG did not converge within the maximum number of sweeps."
    return prev_energy, mps
end
