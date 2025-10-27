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

# Initialize only right environments
function initialize_right_environments!(cache::EnvironmentCache, H::MPO, mps::MPS)
    N = mps.N
    
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

    # Contract H_i and H_i+1 first
    # H_i[mpo_L, phys_i_in, phys_i_out, mpo_mid]
    # H_iplus1[mpo_mid, phys_iplus1_in, phys_iplus1_out, mpo_R]
    # Result: H_two[mpo_L, phys_i_in, phys_i_out, phys_iplus1_in, phys_iplus1_out, mpo_R]
    H_two = zeros(Complex{Float64}, size(H_i, 1), size(H_i, 2), size(H_i, 3), size(H_iplus1, 2), size(H_iplus1, 3), size(H_iplus1, 4))
    @tensor H_two[a, b, c, d, e, f] = H_i[a, b, c, g] * H_iplus1[g, d, e, f]

    # Contract with left environment
    # L[mps_L_bra, mpo_L, mps_L_ket]
    # Result: temp1[mps_L_bra, mps_L_ket, phys_i_in, phys_i_out, phys_iplus1_in, phys_iplus1_out, mpo_R]
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] = L[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    # R[mps_R_bra, mpo_R, mps_R_ket]
    # Result: temp2[mps_L_bra, mps_L_ket, phys_i_in, phys_i_out, phys_iplus1_in, phys_iplus1_out, mps_R_bra, mps_R_ket]
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] = temp1[a, b, c, d, e, f, i] * R[g, i, h]

    # Reshape temp2 to construct the effective Hamiltonian H_eff
    # temp2[mps_L_bra, mps_L_ket, phys_i_in, phys_i_out, phys_iplus1_in, phys_iplus1_out, mps_R_bra, mps_R_ket]
    # Need to reshape to H[bra_indices, ket_indices] where:
    # - bra_indices = [mps_L_bra, phys_i_out, phys_{i+1}_out, mps_R_bra]
    # - ket_indices = [mps_L_ket, phys_i_in, phys_{i+1}_in, mps_R_ket]
    chi_L = size(temp2, 1)  # mps_left bond (bra)
    chi_R = size(temp2, 7)  # mps_right bond (bra)
    # Permute to [mps_L_bra, phys_i_out, phys_{i+1}_out, mps_R_bra, mps_L_ket, phys_i_in, phys_{i+1}_in, mps_R_ket]
    temp2_perm = permutedims(temp2, [1, 4, 6, 7, 2, 3, 5, 8])
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
    # H_im1[mpo_L, phys_im1_in, phys_im1_out, mpo_mid]
    # H_i[mpo_mid, phys_i_in, phys_i_out, mpo_R]
    # Result: H_two[mpo_L, phys_im1_in, phys_im1_out, phys_i_in, phys_i_out, mpo_R]
    H_two = zeros(Complex{Float64}, size(H_im1, 1), size(H_im1, 2), size(H_im1, 3), size(H_i, 2), size(H_i, 3), size(H_i, 4))
    @tensor H_two[a, b, c, d, e, f] = H_im1[a, b, c, g] * H_i[g, d, e, f]
    
    # Contract with left environment
    # L[mps_L_bra, mpo_L, mps_L_ket]
    # Result: temp1[mps_L_bra, mps_L_ket, phys_im1_in, phys_im1_out, phys_i_in, phys_i_out, mpo_R]
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] = L[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    # R[mps_R_bra, mpo_R, mps_R_ket]
    # Result: temp2[mps_L_bra, mps_L_ket, phys_im1_in, phys_im1_out, phys_i_in, phys_i_out, mps_R_bra, mps_R_ket]
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] = temp1[a, b, c, d, e, f, i] * R[g, i, h]
    
    # Reshape temp2 to construct the effective Hamiltonian H_eff
    # temp2[mps_L_bra, mps_L_ket, phys_{i-1}_in, phys_{i-1}_out, phys_i_in, phys_i_out, mps_R_bra, mps_R_ket]
    # Need to reshape to H[bra_indices, ket_indices] where:
    # - bra_indices = [mps_L_bra, phys_{i-1}_out, phys_i_out, mps_R_bra]
    # - ket_indices = [mps_L_ket, phys_{i-1}_in, phys_i_in, mps_R_ket]
    chi_L = size(temp2, 1)  # mps_left bond (bra)
    chi_R = size(temp2, 7)  # mps_right bond (bra)
    # Permute to [mps_L_bra, phys_{i-1}_out, phys_i_out, mps_R_bra, mps_L_ket, phys_{i-1}_in, phys_i_in, mps_R_ket]
    temp2_perm = permutedims(temp2, [1, 4, 6, 7, 2, 3, 5, 8])
    H_eff = reshape(temp2_perm, (chi_L * d * d * chi_R, chi_L * d * d * chi_R))

    return H_eff
end




# Function to solve the eigenvalue problem
function eigsolve(H_eff::Matrix{ComplexF64}, init_state::Vector{ComplexF64}, num_eigvals::Int, which_eigvals::Symbol)
    # Use KrylovKit's eigsolve function with improved parameters for better convergence
    vals, vecs, info = KrylovKit.eigsolve(H_eff, init_state, num_eigvals, which_eigvals;
                                          tol=1e-10,          # Tighter tolerance
                                          krylovdim=30,       # Larger Krylov subspace
                                          maxiter=200,        # More iterations allowed
                                          ishermitian=true)   # Hamiltonian should be Hermitian
    return vals[1], vecs[1]  # Return the ground state energy and wavefunction
end


function dmrg_sweep!(H::MPO, mps::MPS, cache::EnvironmentCache, direction::Symbol, χ_max::Int, tol::Float64, hubbard=false)
    energy = 0.0
    # For two-site DMRG, we optimize pairs of sites
    # Right sweep: optimize (i, i+1) for i in 1:N-1
    # Left sweep: optimize (i-1, i) for i in N:-1:2, which is equivalent to pairs (i, i+1) for i in (N-1):-1:1
    range = direction == :right ? (1:mps.N-1) : reverse(2:mps.N)
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
        
        # Ground state from eigsolve is already normalized, no need to normalize again
        
        # Perform SVD and truncate
        F = svd(ground_state)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncate bond dimension: keep at most χ_max singular values above a small threshold
        # We use a much smaller threshold than tol (energy convergence) to avoid losing important states
        svd_threshold = 1e-14  # Keep singular values above machine precision
        χ_trunc = min(χ_max, length(S), count(>(svd_threshold), S))
        # Ensure at least one singular value is kept
        χ_trunc = max(1, χ_trunc)
        
        # Calculate truncation error before truncation
        truncation_error = χ_trunc < length(S) ? sum(abs2(S[i]) for i in (χ_trunc+1):length(S)) : 0.0
        total_truncation_error += truncation_error
        
        # Truncate arrays
        U = @view U[:, 1:χ_trunc]
        S_trunc = @view S[1:χ_trunc]
        Vt = @view Vt[1:χ_trunc, :]
        
        # Update MPS tensors
        if direction == :right
            mps.tensors[i] = reshape(U, (chi_i_left, mps.d, χ_trunc))
            mps.tensors[i+1] = reshape(Diagonal(S_trunc) * Vt, (χ_trunc, mps.d, chi_iplus1_right))
            # Reinitialize entire cache after updating tensors to ensure consistency
            initialize_cache!(cache, H, mps)
        else
            mps.tensors[i-1] = reshape(U, (chi_iminus1_left, mps.d, χ_trunc))
            mps.tensors[i] = reshape(Diagonal(S_trunc) * Vt, (χ_trunc, mps.d, chi_i_right))
            # Reinitialize entire cache after updating tensors to ensure consistency
            initialize_cache!(cache, H, mps)
            # Update left environment after moving the orthogonality center
            # L[i] depends on mps.tensors[i] and L[i-1]
            # L[i+1] depends on mps.tensors[i+1] and L[i]
            # After updating both tensors, we must update L[i] first (using the unchanged L[i-1]),
            # then update L[i+1] (using the freshly computed L[i])
            update_left_environment!(cache, H, mps, i)
            if i+1 < mps.N
                update_left_environment!(cache, H, mps, i+1)
            end
        else
            mps.tensors[i-1] = reshape(U * Diagonal(S_trunc), (chi_iminus1_left, mps.d, χ_trunc))
            mps.tensors[i] = reshape(Vt, (χ_trunc, mps.d, chi_i_right))
            # Update right environment after moving the orthogonality center
            # R[i] depends on mps.tensors[i] and R[i+1]
            # R[i-1] depends on mps.tensors[i-1] and R[i]
            # After updating both tensors, we must update R[i] first (using the unchanged R[i+1]),
            # then update R[i-1] (using the freshly computed R[i])
            if i < mps.N
                update_right_environment!(cache, H, mps, i+1)  # Updates R[i] using R[i+1]
            end
            update_right_environment!(cache, H, mps, i)  # Updates R[i-1] using updated R[i]
        end
    end
    
    return real(energy), total_truncation_error, mps
end

# Function to compute the expectation value of the Hamiltonian with an MPS
function compute_energy(H::MPO, mps::MPS)
    N = mps.N
    
    # Contract: <mps|H|mps>
    # Start from the left with a 1x1x1 tensor
    E = ones(Complex{Float64}, 1, 1, 1)
    
    for i in 1:N
        mps_i = mps.tensors[i]
        mps_i_conj = conj(mps_i)
        H_i = H.tensor[i]
        
        # Contract: E[bond_in, mpo_bond_in, bond_in'] * 
        #           mps_i[bond_in, phys, bond_out] *
        #           H_i[mpo_bond_in, phys, phys', mpo_bond_out] *
        #           mps_i_conj[bond_in', phys', bond_out']
        # Result: E_new[bond_out, mpo_bond_out, bond_out']
        
        E_temp = zeros(Complex{Float64}, size(mps_i, 3), size(H_i, 4), size(mps_i, 3))
        @einsum E_temp[a, b, c] = E[χ1, χ2, χ3] * mps_i[χ1, d1, a] * H_i[χ2, d1, d2, b] * mps_i_conj[χ3, d2, c]
        E = E_temp
    end
    
    # Final result should be a 1x1x1 tensor
    return real(E[1,1,1])
end


function dmrg(H::MPO, mps::MPS, max_sweeps::Int, χ_max::Int, tol::Float64, hubbard=false)
    energy = 0.0
    prev_energy = 0.0
    
    # Put MPS in right-canonical form initially
    right_normalize!(mps)
    
    # Initialize environment cache once at the start
    cache = EnvironmentCache()
    
    for sweep in 1:max_sweeps
        # Right sweep (cache is reinitialized within dmrg_sweep! after each tensor update)
        energy_right, trunc_error_right, mps = dmrg_sweep!(H, mps, cache, :right, χ_max, tol, hubbard)
        
        # Left sweep (cache is reinitialized within dmrg_sweep! after each tensor update)
        # Rebuild environments at start of each full sweep (right + left)
        initialize_cache!(cache, H, mps)
        
        # Right sweep
        energy_right, trunc_error_right, mps = dmrg_sweep!(H, mps, cache, :right, χ_max, tol, hubbard)
        
        # Rebuild environments before left sweep
        initialize_cache!(cache, H, mps)
        
        # Left sweep
        energy_left, trunc_error_left, mps = dmrg_sweep!(H, mps, cache, :left, χ_max, tol, hubbard)
        
        # Total truncation error for this sweep
        total_truncation_error = trunc_error_right + trunc_error_left
        
        # Compute the actual energy of the MPS (expectation value of H)
        current_energy = compute_energy(H, mps)
        
        # Check for convergence based on energy change
        energy_change = abs(current_energy - prev_energy)
        
        @printf("Sweep %d completed. Energy = %.12f, Energy Change = %.12e, Truncation Error = %.12e\n", 
                sweep, current_energy, energy_change, total_truncation_error)
        
        # Check convergence
        if sweep > 1 && energy_change < tol
            @printf("Converged after %d sweeps. Final Energy = %.12f, Final Truncation Error = %.12e\n", 
                    sweep, current_energy, total_truncation_error)
            return current_energy, mps
        end
        
        # Update energy
        prev_energy = current_energy
    end
    
    @warn "DMRG did not converge within the maximum number of sweeps."
    return prev_energy, mps
end


# ============================================================================
# Single-Site DMRG Implementation
# ============================================================================

"""
    construct_effective_hamiltonian_single_site(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)

Construct the effective Hamiltonian for a single-site DMRG optimization using cached environments.

This function builds the effective Hamiltonian for site i in a Matrix Product State (MPS) 
with respect to a Matrix Product Operator (MPO) Hamiltonian, using pre-computed left and right environments.

# Arguments
- `cache::EnvironmentCache`: Cache containing pre-computed environments
- `H::MPO`: The Hamiltonian in MPO form
- `mps::MPS`: The current Matrix Product State
- `i::Int`: The index of the site to optimize

# Returns
- `H_eff::Matrix{Complex{Float64}}`: The effective Hamiltonian as a matrix
"""
function construct_effective_hamiltonian_single_site(cache::EnvironmentCache, H::MPO, mps::MPS, i::Int)
    # Get left and right environments from cache
    L = cache.L[i-1]
    R = cache.R[i]

    d = size(mps.tensors[i], 2)
    H_i = H.tensor[i]

    # Contract with left environment
    # L[chi_L_bra, mpo_L, chi_L_ket] * H_i[mpo_L, phys_in, phys_out, mpo_R]
    # Result: temp1[chi_L_bra, chi_L_ket, phys_in, phys_out, mpo_R]
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_i, 2), size(H_i, 3), size(H_i, 4))
    @tensor temp1[a, b, c, d, e] = L[a, f, b] * H_i[f, c, d, e]

    # Contract with right environment
    # temp1[chi_L_bra, chi_L_ket, phys_in, phys_out, mpo_R] * R[chi_R_bra, mpo_R, chi_R_ket]
    # Result: temp2[chi_L_bra, chi_L_ket, phys_in, phys_out, chi_R_bra, chi_R_ket]
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f] = temp1[a, b, c, d, g] * R[e, g, f]

    # Reshape temp2 to construct the effective Hamiltonian H_eff
    # temp2[chi_L_bra, chi_L_ket, phys_in, phys_out, chi_R_bra, chi_R_ket]
    # Need to reshape to H[bra_indices, ket_indices] where:
    # - bra_indices = [chi_L_bra, phys_out, chi_R_bra]
    # - ket_indices = [chi_L_ket, phys_in, chi_R_ket]
    chi_L = size(temp2, 1)
    chi_R = size(temp2, 5)
    # Permute to [chi_L_bra, phys_out, chi_R_bra, chi_L_ket, phys_in, chi_R_ket]
    temp2_perm = permutedims(temp2, [1, 4, 5, 2, 3, 6])
    H_eff = reshape(temp2_perm, (chi_L * d * chi_R, chi_L * d * chi_R))

    return H_eff
end


"""
    dmrg_sweep_single_site!(H::MPO, mps::MPS, cache::EnvironmentCache, direction::Symbol)

Perform a single DMRG sweep using single-site optimization.

In single-site DMRG, we optimize one site at a time without changing bond dimensions.
This is faster than two-site DMRG but cannot increase bond dimensions during optimization.

# Arguments
- `H::MPO`: The Hamiltonian MPO
- `mps::MPS`: The current MPS (modified in-place)
- `cache::EnvironmentCache`: Environment cache
- `direction::Symbol`: Either `:right` (left-to-right sweep) or `:left` (right-to-left sweep)

# Returns
- `energy::Float64`: The energy from the last site optimization
- `mps::MPS`: The updated MPS
"""
function dmrg_sweep_single_site!(H::MPO, mps::MPS, cache::EnvironmentCache, direction::Symbol)
    energy = 0.0
    range = direction == :right ? (1:mps.N) : reverse(1:mps.N)
    
    for i in range
        # Construct effective Hamiltonian for single site
        H_eff = construct_effective_hamiltonian_single_site(cache, H, mps, i)
        
        # Current site tensor as a vector
        site_tensor = mps.tensors[i]
        chi_left = size(site_tensor, 1)
        chi_right = size(site_tensor, 3)
        site_vec = vec(site_tensor)
        
        # Solve for ground state using eigsolve
        energy, ground_state = eigsolve(H_eff, site_vec, 1, :SR)
        
        # Reshape ground state back to a tensor
        mps.tensors[i] = reshape(ground_state, (chi_left, mps.d, chi_right))
        
        # Update environments after modifying the tensor
        if direction == :right
            # Right sweep: update left environment
            if i < mps.N
                update_left_environment!(cache, H, mps, i)
            end
        else
            # Left sweep: update right environment
            if i > 1
                update_right_environment!(cache, H, mps, i)
            end
        end
    end
    
    return real(energy), mps
end


"""
    dmrg_single_site(H::MPO, mps::MPS, max_sweeps::Int, tol::Float64)

Perform DMRG optimization using single-site algorithm.

Single-site DMRG optimizes one site at a time without changing bond dimensions.
This is an alternative to two-site DMRG that is faster but cannot dynamically 
adjust bond dimensions during optimization.

# Arguments
- `H::MPO`: The Hamiltonian in MPO form
- `mps::MPS`: Initial MPS guess (bond dimensions are fixed)
- `max_sweeps::Int`: Maximum number of DMRG sweeps
- `tol::Float64`: Convergence tolerance for energy change

# Returns
- `energy::Float64`: The ground state energy
- `mps::MPS`: The optimized MPS

# Notes
- The bond dimensions of the MPS are not changed during optimization
- For increasing bond dimensions, use the two-site `dmrg` function instead
- Single-site DMRG is typically faster per sweep than two-site DMRG
- Good for refinement after initial two-site DMRG optimization

# Example
```julia
# Create Hamiltonian and initial MPS with fixed bond dimension
H = heisenberg_ham(N, d, χ_mpo)
mps = random_MPS(N, d, χ_mps)

# Optimize with single-site DMRG
energy, optimized_mps = dmrg_single_site(H, mps, 50, 1e-8)
```
"""
function dmrg_single_site(H::MPO, mps::MPS, max_sweeps::Int, tol::Float64)
    energy = 0.0
    prev_energy = 0.0
    
    # NOTE: We do NOT normalize the MPS here to preserve the state from two-site DMRG
    # Single-site DMRG works best when used for refinement after two-site DMRG
    # and should use the MPS as-is to avoid changing bond dimensions
    
    # Initialize environment cache once at the start
    cache = EnvironmentCache()
    
    for sweep in 1:max_sweeps
        # Rebuild environments at start of each full sweep (right + left)
        initialize_cache!(cache, H, mps)
        
        # Right sweep
        energy_right, mps = dmrg_sweep_single_site!(H, mps, cache, :right)
        
        # Rebuild environments before left sweep
        initialize_cache!(cache, H, mps)
        
        # Left sweep
        energy_left, mps = dmrg_sweep_single_site!(H, mps, cache, :left)
        
        # Compute the actual energy of the MPS (expectation value of H)
        current_energy = compute_energy(H, mps)
        
        # Check for convergence based on energy change
        energy_change = abs(current_energy - prev_energy)
        
        @printf("Single-site DMRG Sweep %d completed. Energy = %.12f, Energy Change = %.12e\n", 
                sweep, current_energy, energy_change)
        
        # Check convergence
        if sweep > 1 && energy_change < tol
            @printf("Single-site DMRG converged after %d sweeps. Final Energy = %.12f\n", 
                    sweep, current_energy)
            return current_energy, mps
        end
        
        # Update energy
        prev_energy = current_energy
    end
    
    @warn "Single-site DMRG did not converge within the maximum number of sweeps."
    return prev_energy, mps
end
