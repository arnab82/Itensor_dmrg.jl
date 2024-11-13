
using LinearAlgebra



# Define 4x4 operators directly, ensuring each is structured for a four-level system
function create_site_operator(op_type::String, d::Int)
    if op_type == "Cdagup"
        # Create Cdagup as a 4x4 matrix
        return [0 0 0 0; 
                0 0 0 0; 
                0 1 0 0; 
                0 0 0 0]
    elseif op_type == "Cup"
        # Create Cup as a 4x4 matrix
        return [0 0 0 0; 
                0 0 1 0; 
                0 0 0 0; 
                0 0 0 0]
    elseif op_type == "Cdagdn"
        # Create Cdagdn as a 4x4 matrix
        return [0 0 0 0; 
                0 0 0 0; 
                0 0 0 0; 
                0 1 0 0]
    elseif op_type == "Cdn"
        # Create Cdn as a 4x4 matrix
        return [0 0 0 0; 
                0 0 0 0; 
                0 0 0 0; 
                0 0 1 0]
    elseif op_type == "Nup"
        # Create Nup as a 4x4 matrix
        return [0 0 0 0; 
                0 1 0 0; 
                0 0 1 0; 
                0 0 0 0]
    elseif op_type == "Ndn"
        # Create Ndn as a 4x4 matrix
        return [0 0 0 0; 
                0 0 0 0; 
                0 0 1 0; 
                0 0 0 1]
    elseif op_type == "Nupdn"
        # Create Nupdn as a 4x4 matrix
        return [0 0 0 0; 
                0 0 0 0; 
                0 0 0 0; 
                0 0 0 1]
    else
        error("Unknown operator type: $op_type")
    end
end

function hubbard_1d(; N::Int, t=1.0, U=0.0)
    d = 4  # Local Hilbert space dimension (empty, up, down, up+down)
    
    # Initialize MPO tensors
    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, N)
    
    # Define operators
    I_ = Matrix{ComplexF64}(I, d, d)
    Cup = create_site_operator("Cup", d)
    Cdagup = create_site_operator("Cdagup", d)
    Cdn = create_site_operator("Cdn", d)
    Cdagdn = create_site_operator("Cdagdn", d)
    Nupdn = create_site_operator("Nupdn", d)
    
    # Construct MPO tensors
    for i in 1:N
        if i == 1
            # First site: (1, d, d, 5)
            mpo_tensors[i] = zeros(ComplexF64, 1, d, d, 5)
            mpo_tensors[i][1,:, :, 1] = I_
            mpo_tensors[i][1,:, :, 2] = -t * Cdagup
            mpo_tensors[i][1,:, :, 3] = -t * Cup
            mpo_tensors[i][1,:, :, 4] = -t * Cdagdn
            mpo_tensors[i][1,:, :, 5] = -t * Cdn
            if U != 0
                mpo_tensors[i][1,:, :, 5] += U * Nupdn
            end
        elseif i == N
            # Last site: (5, d, d, 1)
            mpo_tensors[i] = zeros(ComplexF64, 5, d, d, 1)
            mpo_tensors[i][1,:, :, 1] = I_
            mpo_tensors[i][2,:, :, 1] = Cup
            mpo_tensors[i][3,:, :, 1] = Cdagup
            mpo_tensors[i][4,:, :, 1] = Cdn
            mpo_tensors[i][5,:, :, 1] = Cdagdn
            if U != 0
                mpo_tensors[i][5,:, :, 1] += U * Nupdn
            end
        else
            # Middle sites: (5, d, d, 5)
            mpo_tensors[i] = zeros(ComplexF64, 5, d, d, 5)
            mpo_tensors[i][1,:, :, 1] = I_
            mpo_tensors[i][2,:, :, 1] = Cup
            mpo_tensors[i][3,:, :, 1] = Cdagup
            mpo_tensors[i][4,:, :, 1] = Cdn
            mpo_tensors[i][5,:, :, 1] = Cdagdn
            mpo_tensors[i][5,:, :, 2] = -t * Cdagup
            mpo_tensors[i][5,:, :, 3] = -t * Cup
            mpo_tensors[i][5,:, :, 4] = -t * Cdagdn
            mpo_tensors[i][5,:, :, 5] = I_
            if U != 0
                mpo_tensors[i][5,:, :, 5] += U * Nupdn
            end
        end
    end
    return MPO(mpo_tensors)
end
      

# For 2D Hubbard model, we'll need to implement a function to map 2D coordinates to 1D
function coord_2d_to_1d(x::Int, y::Int, Ny::Int)
    return (x - 1) * Ny + y
end

function hubbard_2d(; Nx::Int, Ny::Int, t=1.0, U=0.0, yperiodic::Bool=true)
    N = Nx * Ny
    d = 4  # Local Hilbert space dimension (empty, up, down, up+down)
    
    # Initialize MPO tensors
    mpo_tensors = Vector{Array{ComplexF64, 4}}(undef, N)
    
    # Define operators
    I_ = Matrix{ComplexF64}(I, d, d)
    Cup = create_site_operator("Cup", d)
    Cdagup = create_site_operator("Cdagup", d)
    Cdn = create_site_operator("Cdn", d)
    Cdagdn = create_site_operator("Cdagdn", d)
    Nupdn = create_site_operator("Nupdn", d)
    
    # Construct MPO tensors
    for x in 1:Nx
        for y in 1:Ny
            i = coord_2d_to_1d(x, y, Ny)
            
            if i == 1
                # First site: (1, d, d, 5)
                mpo_tensors[i] = zeros(ComplexF64, 1, d, d, 5)
                mpo_tensors[i][1,:, :, 1] = I_
                mpo_tensors[i][1,:, :, 2] = -t * Cdagup
                mpo_tensors[i][1,:, :, 3] = -t * Cup
                mpo_tensors[i][1,:, :, 4] = -t * Cdagdn
                mpo_tensors[i][1,:, :, 5] = -t * Cdn
                if U != 0
                    mpo_tensors[i][1,:, :, 5] += U * Nupdn
                end
            elseif i == N
                # Last site: (5, d, d, 1)
                mpo_tensors[i] = zeros(ComplexF64, 5, d, d, 1)
                mpo_tensors[i][1,:, :, 1] = I_
                mpo_tensors[i][2,:, :, 1] = Cup
                mpo_tensors[i][3,:, :, 1] = Cdagup
                mpo_tensors[i][4,:, :, 1] = Cdn
                mpo_tensors[i][5,:, :, 1] = Cdagdn
                if U != 0
                    mpo_tensors[i][5,:, :, 1] += U * Nupdn
                end
            else
                # Middle sites: (5, d, d, 5)
                mpo_tensors[i] = zeros(ComplexF64, 5, d, d, 5)
                mpo_tensors[i][1,:, :, 1] = I_
                mpo_tensors[i][2,:, :, 1] = Cup
                mpo_tensors[i][3,:, :, 1] = Cdagup
                mpo_tensors[i][4,:, :, 1] = Cdn
                mpo_tensors[i][5,:, :, 1] = Cdagdn
                mpo_tensors[i][5,:, :, 2] = -t * Cdagup
                mpo_tensors[i][5,:, :, 3] = -t * Cup
                mpo_tensors[i][5,:, :, 4] = -t * Cdagdn
                mpo_tensors[i][5,:, :, 5] = I_
                if U != 0
                    mpo_tensors[i][5,:, :, 5] += U * Nupdn
                end
            end
        end
    end
    
    return MPO(mpo_tensors)
end


function hubbard(; Nx::Int, Ny::Int=1, t=1.0, U=0.0, yperiodic::Bool=true, ky::Bool=false)
    if Ny == 1
        return hubbard_1d(; N=Nx, t=t, U=U)
    elseif ky
        error("hubbard_2d_ky not implemented in this version")
    else
        return hubbard_2d(; Nx=Nx, Ny=Ny, t=t, U=U, yperiodic=yperiodic)
    end
end
