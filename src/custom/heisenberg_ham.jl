

# Helper function to create single-site operators
function create_site_operator_heisenberg(op_type::String)
    if op_type == "S+"
        return [0 1; 0 0]
    elseif op_type == "S-"
        return [0 0; 1 0]
    elseif op_type == "Sz"
        return [0.5 0; 0 -0.5]
    elseif op_type == "I"
        return [1 0; 0 1]
    else
        error("Unknown operator type: $op_type")
    end
end
""" 
    heisenberg_ham(N::Int)
    Args:
        N: Number of sites
    Returns:
        MPO: Heisenberg Hamiltonian as an MPO

        H = Σ_i [0.5 * (S+i * S-{i+1} + S-i * S+{i+1}) + Sz_i * Sz_{i+1}]

"""
function heisenberg_ham(N::Int,d::Int,χ::Int)
    # d = 2  # Local Hilbert space dimension (spin up, spin down)
    # χ = 5  # Bond dimension for internal MPO bonds

    # Initialize MPO tensors
    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, N)

    # Define operators
    I = create_site_operator_heisenberg("I")
    Sp = create_site_operator_heisenberg("S+")
    Sm = create_site_operator_heisenberg("S-")
    Sz = create_site_operator_heisenberg("Sz")
    Zero = zeros(ComplexF64, d, d)

    # Construct MPO tensors
    # MPO structure for nearest-neighbor Heisenberg:
    # Bond index structure: [I, Sm, Sp, Sz, H_accumulated]
    # where H_accumulated accumulates all completed two-site terms
    
    for i in 1:N
        if i == 1
            # First site: [H_accumulated, 0.5*S+, 0.5*S-, Sz, I]
            # H_accumulated starts at 0 (no terms completed yet)
            mpo_tensors[i] = zeros(ComplexF64, 1, d, d, χ)
            mpo_tensors[i][1,:, :,1] = Zero         # No accumulated Hamiltonian yet
            mpo_tensors[i][1,:, :,2] = 0.5 * Sp    # Start S+ chain
            mpo_tensors[i][1,:, :,3] = 0.5 * Sm    # Start S- chain
            mpo_tensors[i][1,:, :,4] = Sz          # Start Sz chain
            mpo_tensors[i][1,:, :,5] = I           # Identity to pass through
        elseif i == N
            # Last site: [I, S-, S+, Sz, H_accumulated]^T
            # All chains must terminate here
            mpo_tensors[i] = zeros(ComplexF64, χ, d, d, 1)
            mpo_tensors[i][1, :, :,1] = I          # Identity from pass-through
            mpo_tensors[i][2,:, :,1] = Sm         # Complete S+ from previous site
            mpo_tensors[i][3,:, :,1] = Sp         # Complete S- from previous site
            mpo_tensors[i][4,:, :,1] = Sz         # Complete Sz from previous site
            mpo_tensors[i][5,:, :,1] = Zero        # H_accumulated (but we don't add it again)
        else
            # Middle sites
            mpo_tensors[i] = zeros(ComplexF64, χ, d, d, χ)
            # Top-left: Identity path (no interaction)
            mpo_tensors[i][1,:, :,1] = I
            # Complete chains from left
            mpo_tensors[i][2,:, :,1] = Sm         # Complete S+[i-1] S-[i]
            mpo_tensors[i][3,:, :,1] = Sp         # Complete S-[i-1] S+[i]
            mpo_tensors[i][4,:, :,1] = Sz         # Complete Sz[i-1] Sz[i]
            # Bottom-left column: pass through H_accumulated
            mpo_tensors[i][5,:, :,1] = Zero
            # Bottom row: start new chains
            mpo_tensors[i][5,:, :,2] = 0.5 * Sp
            mpo_tensors[i][5,:, :,3] = 0.5 * Sm
            mpo_tensors[i][5,:, :,4] = Sz
            mpo_tensors[i][5,:, :,5] = I
        end
    end

    return MPO(mpo_tensors)
end
