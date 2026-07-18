using LinearAlgebra
using Statistics
using NPZ

"""
Calculate the direct product of two tensors.
Optimized version with reduced allocations.
"""
function composetensors(A::Array{Complex{Float64},4}, B::Array{Complex{Float64},4})
    d1A, d2A, rA, cA = size(A)
    d1B, d2B, rB, cB = size(B)
    AB = zeros(Complex{Float64}, (d1A, d2B, rA*rB, cA*cB))
    
    # Use @inbounds for performance in this verified loop
    @inbounds for z1 in 1:d1A, z2 in 1:d2B, z in 1:d2A
        # Direct kron computation, avoiding intermediate allocation
        for i in 1:rA, j in 1:rB, k in 1:cA, l in 1:cB
            AB[z1, z2, (i-1)*rB + j, (k-1)*cB + l] += A[z1, z, i, k] * B[z, z2, j, l]
        end
    end
    return AB
end

"""
Calculate the direct sum of two tensors.
"""
function combinetensors(A::Array{Complex{Float64},4}, B::Array{Complex{Float64},4})
    d1A, d2A, rA, cA = size(A)
    d1B, d2B, rB, cB = size(B)
    AB = zeros(Complex{Float64}, (d1A, d2B, rA+rB, cA+cB))
    # Use views to avoid copying
    AB[:, :, 1:rA, 1:cA] .= A
    AB[:, :, rA+1:end, cA+1:end] .= B
    return AB
end

"""
Perform singular value decomposition on a matrix A.
"""
function decompose(A::Matrix{Complex{Float64}}; R::Union{Int,Nothing}=nothing, ε::Float64=0.0, ζ::Float64=0.0)
    U, s, V = svd(A)
    if isnothing(R) || R > length(s)
        R = length(s)
    end
    
    snorm = cumsum(s) / sum(s)
    for (i, tab) in enumerate(snorm)
        if i >= R
            break
        end
        if tab >= 1 - ε
            R = i + 1
        end
        if i > 0 && s[i] / s[i-1] < ζ
            R = i
        end
    end
    
    return U[:,1:R], Diagonal(s[1:R]), V'[1:R,:]
end


"""
constructor for MPO
    args:
        tensor: Vector{Array{Complex{Float64},4}}
        d1: Int
        d2: Int
        N: Int
        k: Int
    return:
        MPO
    
"""


mutable struct MPO
    tensor::Vector{Array{Complex{Float64},4}}
    N::Int
    d1::Int
    d2::Int
    k::Int

    function MPO(tensor::Vector{Array{Complex{Float64},4}})
        N = length(tensor)
        # d1 and d2 are the physical dimensions (indices 2 and 3 in the tensor)
        # not the bond dimensions (indices 1 and 4)
        d1 = N == 0 ? 0 : size(tensor[1], 2)
        d2 = N == 0 ? 0 : size(tensor[1], 3)
        new(tensor, N, d1, d2, -1)
    end
end

Base.show(io::IO, mpo::MPO) = print(io, join(["Site $n:\n$(mpo.tensor[n])" for n in 1:mpo.N], "\n" * "-"^20 * "\n"))

Base.copy(mpo::MPO) = MPO(deepcopy(mpo.tensor))


"""
    Multiply two MPOs.

"""
function Base.:*(A::MPO, B::MPO)
    tensor = [composetensors(B.tensor[n], A.tensor[n]) for n in 1:A.N]
    return MPO(tensor)
end

"""
    Add two MPOs.

"""

function Base.:+(A::MPO, B::MPO)
    tensor = Vector{Array{Complex{Float64},4}}(undef, A.N)
    tensor[1] = cat(A.tensor[1], B.tensor[1], dims=4)
    for n in 2:A.N-1
        tensor[n] = combinetensors(A.tensor[n], B.tensor[n])
    end
    tensor[end] = cat(A.tensor[end], B.tensor[end], dims=3)
    return MPO(tensor)
end

"""
    multiply a MPO by a scalar
"""

function Base.:*(c::Number, mpo::MPO)
    O = copy(mpo)
    if O.k < 0
        O.tensor[1] .*= c
    else
        O.tensor[O.k] .*= c
    end
    return O
end

Base.:*(mpo::MPO, c::Number) = c * mpo

"""
Calculate the dimension of the MPO.

"""

function dimension(mpo::MPO)
    return maximum(maximum(size(site)[3:4]) for site in mpo.tensor)
end

"""
Calculate the bond dimensions of the MPO.
"""

function bonds(mpo::MPO)
    bonds = [size(site, 3) for site in mpo.tensor]
    push!(bonds, 1)
    return bonds
end

"""
Calculate the dual of the MPO.
"""

function dual(mpo::MPO)
    tensor = [permutedims(conj(site), (2,1,3,4)) for site in mpo.tensor]
    return MPO(tensor)
end

"""
Calculate the inner product of two MPOs.
"""

function inner(A::MPO, B::MPO)
    train = A * dual(B)
    M = Matrix{Complex{Float64}}(I, 1, 1)
    for n in 1:A.N
        S = sum(train.tensor[n][z,z,:,:] for z in 1:A.d2)
        M = M * S
    end
    return sum(M)
end

"""
Calculate the norm of the MPO.
"""


function LinearAlgebra.norm(mpo::MPO)
    if mpo.k < 0
        return sqrt(real(inner(mpo, mpo)))
    else
        return sqrt(sum(abs2, mpo.tensor[mpo.k]))
    end
end

"""
Convert the MPO to a matrix.
"""


function MPO_to_array(mpo::MPO)
    # Physical dimensions
    d1 = mpo.d1
    d2 = mpo.d2
    r, c = d1^mpo.N, d2^mpo.N
    O = zeros(Complex{Float64}, (r, c))
    for i in 0:r-1
        ii = digits(i, base=d1, pad=mpo.N)
        for j in 0:c-1
            jj = digits(j, base=d2, pad=mpo.N)
            # Contract MPO tensors along bond dimensions
            # Start with left boundary (1x1 identity)
            co = ones(Complex{Float64}, 1, 1)
            for n in 1:mpo.N
                # Extract the relevant matrix slice [bond_left, bond_right] for physical indices ii[n]+1, jj[n]+1
                mat_slice = mpo.tensor[n][:, ii[n]+1, jj[n]+1, :]
                # Contract: co[bond_in] * mat_slice[bond_in, bond_out] -> co[bond_out]
                co = co * mat_slice
            end
            # After contracting all sites, co should be 1x1
            O[i+1,j+1] = co[1,1]
        end
    end
    return O
end
"""
Calculate the diagonal of the MPO.

"""


function diag(mpo::MPO)
    if mpo.d1 != mpo.d2
        error("diag not defined when d1 != d2")
    end
    r = mpo.d1^mpo.N
    O = zeros(Complex{Float64}, r)
    for i in 0:r-1
        ii = digits(i, base=mpo.d1, pad=mpo.N)
        co = ones(Complex{Float64}, 1, 1)
        for n in 1:mpo.N
            co = co * mpo.tensor[n][ii[n]+1,ii[n]+1,:,:]
        end
        O[i+1] = co[1,1]
    end
    return O
end
"""


"""


function canonize!(mpo::MPO)
    for n in mpo.k+1:mpo.N-1
        A = reduce(vcat, reduce(vcat, mpo.tensor[n]))
        U, S, Vt = decompose(A)
        mpo.tensor[n] = reshape(U, (mpo.d1, mpo.d2, :, size(U, 2)))
        mpo.k += 1
        mpo.tensor[n+1] = reshape((S * Vt) * reshape(mpo.tensor[n+1], (size(Vt, 1), :)),
                                  (mpo.d1, mpo.d2, size(S, 1), :))
    end
end

function normalize!(mpo::MPO)
    norm_val = norm(mpo)
    if norm_val < eps(Float64)
        error("MPO is too close to zero: unnormalizable.")
    end
    mpo *= (1 / sqrt(norm_val))
end

function compress!(mpo::MPO; R::Union{Int,Nothing}=nothing, ε::Float64=0.0, ζ::Float64=0.0)
    if mpo.k < mpo.N - 1
        canonize!(mpo)
    end
    for n in mpo.N:-1:2
        A = reduce(hcat, reduce(vcat, mpo.tensor[n]))
        U, S, Vt = decompose(A; R=R, ε=ε, ζ=ζ)
        mpo.tensor[n-1] = reshape(reshape(mpo.tensor[n-1], (:, mpo.d2)) * (U * S),
                                  (mpo.d1, mpo.d2, :, size(U, 2)))
        mpo.tensor[n] = reshape(Vt, (mpo.d1, mpo.d2, size(Vt, 1), :))
        mpo.k -= 1
    end
end

function save(mpo::MPO, filename::String)
    hexits = ceil(Int, log(mpo.N-1) / log(16))
    tensordict = Dict{String, Array{Complex{Float64}}}()
    for n in 1:mpo.N
        label = lpad(string(n-1, base=16), hexits, '0')
        tensordict[label] = mpo.tensor[n]
    end
    tensordict["N"] = [mpo.N]
    tensordict["k"] = [mpo.k]
    npzwrite(filename, tensordict)
end

function load(filename::String)
    data = npzread(filename)
    N = Int(data["N"][1])
    k = Int(data["k"][1])
    tensor = Vector{Array{Complex{Float64}, 4}}(undef, N)
    for n in 1:N
        label = lpad(string(n-1, base=16), ceil(Int, log(N-1) / log(16)), '0')
        tensor[n] = data[label]
    end
    mpo = MPO(tensor)
    mpo.k = k
    return mpo
end

# VectorTrain struct and methods
struct VectorTrain 
    tensor::Vector{Array{Complex{Float64}, 4}}
    N::Int
    d::Int
    k::Int
end

function VectorTrain(vectors::Matrix{Complex{Float64}})
    N, d = size(vectors)
    tensor = [reshape(vectors[n, :], (d, 1, 1, 1)) for n in 1:N]
    return VectorTrain(tensor, N, d, -1)
end

function entropy(vt::VectorTrain)
    vectors = [vt.tensor[n][:, 1, 1, 1] for n in 1:vt.N]
    return sum(entropy(abs2.(v)) for v in vectors)
end

function vectors(vt::VectorTrain)
    return [vt.tensor[n][:, 1, 1, 1] for n in 1:vt.N]
end