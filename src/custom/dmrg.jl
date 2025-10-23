using LinearAlgebra
using KrylovKit
using Einsum
using TensorOperations
using Printf

# Function to contract two adjacent sites in an MPS (right-moving)
function contract_two_sites_right(mps::MPS, i::Int)
    left_tensor = mps.tensors[i]
    right_tensor = mps.tensors[i+1]
    # println("size of left tensor is ",size(left_tensor))
    # println("size of right tensor is ",size(right_tensor))
    # Create the two-site tensor by contracting along the shared bond dimension
    two_site_tensor = zeros(Complex{Float64}, size(left_tensor, 1), mps.d, mps.d, size(right_tensor, 3))
    @tensor two_site_tensor[l, s1, s2, r] := left_tensor[l, s1, k] * right_tensor[k, s2, r]
    
    # Reshape into a matrix
    chi_left = size(left_tensor, 1)
    chi_right = size(right_tensor, 3)
    return reshape(two_site_tensor, (chi_left * mps.d, mps.d * chi_right))
end

# Function to contract two adjacent sites in an MPS (left-moving)
function contract_two_sites_left(mps::MPS, i::Int)
    left_tensor = mps.tensors[i-1]
    right_tensor = mps.tensors[i]
    
    # Create the two-site tensor by contracting along the shared bond dimension
    two_site_tensor = zeros(Complex{Float64}, size(left_tensor, 1), mps.d, mps.d, size(right_tensor, 3))
    @tensor two_site_tensor[l, s1, s2, r] := left_tensor[l, s1, k] * right_tensor[k, s2, r]
    
    # Reshape into a matrix
    chi_left = size(left_tensor, 1)
    chi_right = size(right_tensor, 3)
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

    construct_effective_hamiltonian(H::MPO, mps::MPS, i::Int)

Construct the effective Hamiltonian for a two-site DMRG optimization.

This function builds the effective Hamiltonian for sites i and i+1 in a Matrix Product State (MPS) 
with respect to a Matrix Product Operator (MPO) Hamiltonian.

# Arguments
- `H::MPO`: The Hamiltonian in MPO form
- `mps::MPS`: The current Matrix Product State
- `i::Int`: The index of the first site of the two-site block

# Returns
- `H_eff::Matrix{Complex{Float64}}`: The effective Hamiltonian as a matrix

Construct the effective Hamiltonian for two adjacent sites in an MPS.
"""

function construct_effective_hamiltonian(H::MPO, mps::MPS, i::Int,chi::Int)
    # Initialize left and right environments
    L = ones(Complex{Float64}, 1, 1, 1)
    R = ones(Complex{Float64}, 1, 1, 1)
    
    # Build left environment
    for j in 1:i-1
        # println(j)
        # println(size(L))
        # println(size(H.tensor[j]))
        # println(size(mps.tensors[j]))
        # Initialize L_temp with correct dimensions for contraction
        mps_i_conj=conj(mps.tensors[j])
        mps_i=mps.tensors[j]
        H_i=H.tensor[j]
        L_temp = zeros(Complex{Float64}, size(mps_i, 1), size(H_i, 2), size(mps_i, 1))
        @einsum L_temp[a, b, c] := L[ χ1,χ2, χ3] * mps_i[χ1,d1,a]* H_i[χ2, d1, d2, b] * mps_i_conj[χ3, d2, c]
        L = L_temp
    end
    # Iterate over each site from the end down to site i+1
    for j in length(H.tensor):-1:i+1
        # Print current dimensions for debugging
        # println("R size: ", size(R))
        # println("H.tensor[j] size: ", size(H.tensor[j]))
        # println("mps.tensors[j] size: ", size(mps.tensors[j]))

        # Initialize R_temp with correct dimensions for contraction
        mps_i = mps.tensors[j]
        mps_i_conj = conj(mps.tensors[j])
        H_i = H.tensor[j]
        R_temp = zeros(Complex{Float64}, size(mps_i, 1), size(H_i, 2), size(mps_i, 1))
        @einsum R_temp[a, b, c] := R[χ1, χ2, χ3] * mps_i[a, d1,χ1 ] * H_i[b, d1, d2, χ2] * mps_i_conj[c, d2, χ3]
        R = R_temp
        # After loop ends, R contains the right environment from site i+1 to the end
    end

    d=size(mps.tensors[i], 2)
    # Initialize effective Hamiltonian
    # H_eff = zeros(Complex{Float64}, size(L, 3) * d^2 * size(R, 3), size(L, 3) * d^2 * size(R, 3))
    H_i=H.tensor[i]
    H_iplus1=H.tensor[i+1]

    # Perform tensor contraction to build the effective Hamiltonian
    #contract L with H_i ,then H_i+1 and then with R
    #contract H_i and H_i+1
    # H_i: [mpo_left, phys_out_1, phys_in_1, mpo_mid]
    # H_iplus1: [mpo_mid, phys_out_2, phys_in_2, mpo_right]
    H_two = zeros(Complex{Float64}, size(H_i, 1), size(H_i, 2), size(H_iplus1, 2), size(H_i, 3), size(H_iplus1, 3), size(H_iplus1, 4))
    @einsum H_two[a, b, c, d, e, f] := H_i[a, b, d, g] * H_iplus1[g, c, e, f]

    # Contract with left environment
    # L: [mps_left, mpo_left, mps_left_conj]
    L_end = L
    temp1 = zeros(Complex{Float64}, size(L_end, 1), size(L_end, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] := L_end[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    # R: [mps_right, mpo_right, mps_right_conj]
    R_start = R
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R_start, 1), size(R_start, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] := temp1[a, b, c, d, e, f, i] * R_start[g, i, h]

    # Reshape temp2 to construct the effective Hamiltonian H_eff
    # temp2 is [mps_left, mps_left_conj, phys_out_1, phys_out_2, phys_in_1, phys_in_2, mps_right, mps_right_conj]
    # We want matrix with:
    # rows (bra): [mps_left_conj, phys_out_1, phys_out_2, mps_right_conj]
    # cols (ket): [mps_left, phys_in_1, phys_in_2, mps_right]
    chi_L = size(temp2, 1)  # mps_left bond (between i-1 and i)
    chi_R = size(temp2, 7)  # mps_right bond (between i+1 and i+2)
    # Permute to: [mps_left_conj, phys_out_1, phys_out_2, mps_right_conj, mps_left, phys_in_1, phys_in_2, mps_right]
    temp2_perm = permutedims(temp2, [2, 3, 4, 8, 1, 5, 6, 7])
    H_eff = reshape(temp2_perm, (chi_L * d * d * chi_R, chi_L * d * d * chi_R))

    # println("Size of H_eff: ", size(H_eff))
    return H_eff
end

function construct_effective_hamiltonian_left(H::MPO, mps::MPS, i::Int, chi::Int)
    # Initialize left and right environments
    L = ones(Complex{Float64}, 1, 1, 1)
    R = ones(Complex{Float64}, 1, 1, 1)
    # println("i is ",i)
    # println("size of MPS is",size(mps.tensors[i]))
    # Build the right environment up to site i-1
    for j in length(H.tensor):-1:i
        # println("Right environment at site ", j)
        # println("R size: ", size(R))
        # println("H.tensor[j] size: ", size(H.tensor[j]))
        # println("mps.tensors[j] size: ", size(mps.tensors[j]))

        mps_j = mps.tensors[j]
        mps_j_conj = conj(mps.tensors[j])
        H_j = H.tensor[j]
        R_temp = zeros(Complex{Float64}, size(mps_j, 1), size(H_j, 2), size(mps_j, 1))
        @einsum R_temp[a, b, c] := R[χ1, χ2, χ3] * mps_j[a, d1, χ1] * H_j[b, d1, d2, χ2] * mps_j_conj[c, d2, χ3]
        R = R_temp
    end

    # Build the left environment up to site i-2
    for j in 1:i-2
        # println("Left environment at site ", j)
        # println("L size: ", size(L))
        # println("H.tensor[j] size: ", size(H.tensor[j]))
        # println("mps.tensors[j] size: ", size(mps.tensors[j]))

        mps_j = mps.tensors[j]
        mps_j_conj = conj(mps.tensors[j])
        H_j = H.tensor[j]
        L_temp = zeros(Complex{Float64}, size(mps_j, 1), size(H_j, 2), size(mps_j, 1))
        @einsum L_temp[a, b, c] := L[χ1, χ2, χ3] * mps_j[χ1, d1, a] * H_j[χ2, d1, d2, b] * mps_j_conj[χ3, d2, c]
        L = L_temp
    end

    # Construct the effective Hamiltonian by contracting L, H_(i-1), H_i, and R
    d = size(mps.tensors[i], 2)
    H_im1 = H.tensor[i - 1]
    H_i = H.tensor[i]

    # Contract H_(i-1) and H_i
    # H_im1: [mpo_left, phys_out_1, phys_in_1, mpo_mid]
    # H_i: [mpo_mid, phys_out_2, phys_in_2, mpo_right]
    H_two = zeros(Complex{Float64}, size(H_im1, 1), size(H_im1, 2), size(H_i, 2), size(H_im1, 3), size(H_i, 3), size(H_i, 4))
    @einsum H_two[a, b, c, d, e, f] := H_im1[a, b, d, g] * H_i[g, c, e, f]
    
    # Contract with left environment
    # L: [mps_left, mpo_left, mps_left_conj]
    temp1 = zeros(Complex{Float64}, size(L, 1), size(L, 3), size(H_two, 2), size(H_two, 3), size(H_two, 4), size(H_two, 5), size(H_two, 6))
    @tensor temp1[a, b, c, d, e, f, g] := L[a, h, b] * H_two[h, c, d, e, f, g]

    # Contract with right environment
    # R: [mps_right, mpo_right, mps_right_conj]
    temp2 = zeros(Complex{Float64}, size(temp1, 1), size(temp1, 2), size(temp1, 3), size(temp1, 4), size(temp1, 5), size(temp1, 6), size(R, 1), size(R, 3))
    @tensor temp2[a, b, c, d, e, f, g, h] := temp1[a, b, c, d, e, f, i] * R[g, i, h]
    
    # Reshape temp2 to construct the effective Hamiltonian H_eff
    # temp2 is [mps_left, mps_left_conj, phys_out_1, phys_out_2, phys_in_1, phys_in_2, mps_right, mps_right_conj]
    # We want matrix with:
    # rows (bra): [mps_left_conj, phys_out_1, phys_out_2, mps_right_conj]
    # cols (ket): [mps_left, phys_in_1, phys_in_2, mps_right]
    chi_L = size(temp2, 1)  # mps_left bond (between i-2 and i-1)
    chi_R = size(temp2, 7)  # mps_right bond (between i and i+1)
    # Permute to: [mps_left_conj, phys_out_1, phys_out_2, mps_right_conj, mps_left, phys_in_1, phys_in_2, mps_right]
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


function dmrg_sweep!(H::MPO, mps::MPS, direction::Symbol, χ_max::Int, tol::Float64,hubbard=false)
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
        # Construct effective Hamiltonian
        if direction == :right
            two_site_tensor = contract_two_sites_right(mps, i)
            H_eff = construct_effective_hamiltonian(H, mps, i,mps.χ)
        else
            two_site_tensor = contract_two_sites_left(mps, i)
            H_eff = construct_effective_hamiltonian_left(H, mps, i,mps.χ)
        end
        # Solve for ground state using eigsolve
        energy, ground_state = eigsolve(H_eff, vec(two_site_tensor), 1, :SR)

        # Reshape ground state back to a tensor (same shape as two_site_tensor)
        ground_state = reshape(ground_state, size(two_site_tensor))
        # Normalize the ground state
        ground_state = ground_state / norm(ground_state)
        # Perform SVD and truncate
        F = svd(ground_state)
        U, S, Vt = F.U, F.S, F.Vt
        # Truncate bond dimension: keep at most χ_max singular values above tolerance
        χ_trunc = min(χ_max, length(S), count(S .> tol))
        # Ensure at least one singular value is kept
        χ_trunc = max(1, χ_trunc)
        truncation_error = sum(abs2, S[χ_trunc+1:end])  # Calculate truncation error
        total_truncation_error += truncation_error  # Accumulate truncation error
        U = U[:, 1:χ_trunc]
        S = S[1:χ_trunc]
        Vt = Vt[1:χ_trunc, :]
        # Update MPS tensors
        if direction == :right
            mps.tensors[i] = reshape(U, (chi_i_left, mps.d, χ_trunc))
            mps.tensors[i+1] = reshape(Diagonal(S) * Vt, (χ_trunc, mps.d, chi_iplus1_right))
        else
            mps.tensors[i-1] = reshape(U, (chi_iminus1_left, mps.d, χ_trunc))
            mps.tensors[i] = reshape(Diagonal(S) * Vt, (χ_trunc, mps.d, chi_i_right))
        end
    end
    
    return real(energy), total_truncation_error , mps
end


function dmrg(H::MPO, mps::MPS, max_sweeps::Int, χ_max::Int, tol::Float64,hubbard=false)
    energy = 0.0
    for sweep in 1:max_sweeps
        # Right sweep
        energy_right, trunc_error_right ,mps= dmrg_sweep!(H, mps, :right, χ_max, tol,hubbard)
        # println("mps_new")
        # display(mps_new)
        # println(size(mps_new.tensors[4]))
        # Left sweep
        energy_left, trunc_error_left,mps = dmrg_sweep!(H, mps, :left, χ_max, tol,hubbard)
        # println(energy_left)
        # Total truncation error for this sweep
        total_truncation_error = trunc_error_right + trunc_error_left
        
        # Check for convergence based on energy change
        println("energy at $sweep ",energy_left,  energy_right)
        energy_change = abs(energy_right- energy)
        avg_sweep_energy = (energy_right + energy_left) / 2
        
        @printf("Sweep %d completed. Energy = %.12f, Energy Change = %.12e, Truncation Error = %.12e\n", 
                sweep, avg_sweep_energy, energy_change, total_truncation_error)
        
        if energy_change < tol
            @printf("Converged after %d sweeps. Final Energy = %.12f, Final Truncation Error = %.12e\n", 
                    sweep, energy_left, total_truncation_error)
            converged = true
            return energy_left, mps
        end
        
        # Update energy
        energy = energy_left
    end
    
    @warn "DMRG did not converge within the maximum number of sweeps."
    return energy, mps
end
