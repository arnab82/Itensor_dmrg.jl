using LinearAlgebra

struct MPS
    tensors::Vector{Array{ComplexF64, 3}}
    N::Int  # Number of sites
    d::Int  # Physical dimension
    χ::Int  # Maximum bond dimension

    function MPS(tensors::Vector{Array{ComplexF64, 3}})
        N = length(tensors)
        d = size(tensors[1], 2)
        χ = maximum(size.(tensors, 3))
        new(tensors, N, d, χ)
    end
end

# Initialize a random MPS
function random_MPS(N::Int, d::Int, χ::Int)
    tensors = [randn(ComplexF64, (i == 1 ? 1 : χ, d, i == N ? 1 : χ)) for i in 1:N]
    return MPS(tensors)
end

# Left-normalize the MPS
function left_normalize!(mps::MPS)
    for i in 1:(mps.N - 1)
        M = reshape(mps.tensors[i], (size(mps.tensors[i], 1) * mps.d, size(mps.tensors[i], 3)))
        F = qr(M)
        mps.tensors[i] = reshape(Matrix(F.Q), (size(mps.tensors[i], 1), mps.d, size(F.Q, 2)))
        mps.tensors[i+1] = reshape(F.R * reshape(mps.tensors[i+1], (size(mps.tensors[i+1], 1), :)), 
                                   (size(F.R, 1), mps.d, size(mps.tensors[i+1], 3)))
    end
end

# Right-normalize the MPS
function right_normalize!(mps::MPS)
    for i in mps.N:-1:2
        M = reshape(mps.tensors[i], (size(mps.tensors[i], 1), size(mps.tensors[i], 2) * size(mps.tensors[i], 3)))
        F = qr(M')
        mps.tensors[i] = reshape(Matrix(F.Q)', (size(F.Q, 2), mps.d, size(mps.tensors[i], 3)))
        mps.tensors[i-1] = reshape(reshape(mps.tensors[i-1], (:, size(mps.tensors[i-1], 3))) * F.R', 
                                   (size(mps.tensors[i-1], 1), mps.d, size(F.R, 1)))
    end
end

# Compute the overlap between two MPS
function overlap(mps1::MPS, mps2::MPS)
    @assert mps1.N == mps2.N
    E = ones(ComplexF64, 1, 1)
    for i in 1:mps1.N
        E = E * reshape(conj.(mps1.tensors[i]), (size(mps1.tensors[i], 1), :)) *
            reshape(mps2.tensors[i], (size(mps2.tensors[i], 1), :))'
    end
    return E[1, 1]
end

# Apply a local operator to a site
function apply_local_operator!(mps::MPS, op::Matrix{ComplexF64}, site::Int)
    @assert size(op) == (mps.d, mps.d)
    mps.tensors[site] = reshape(reshape(mps.tensors[site], (size(mps.tensors[site], 1) * mps.d, :)) * 
                                op', (size(mps.tensors[site], 1), mps.d, size(mps.tensors[site], 3)))
end

# Truncate bond dimension
function truncate!(mps::MPS, χ_max::Int)
    left_normalize!(mps)
    for i in (mps.N - 1):-1:1
        M = reshape(mps.tensors[i], (size(mps.tensors[i], 1) * mps.d, size(mps.tensors[i], 3)))
        U, S, V = svd(M)
        χ = min(χ_max, count(S .> 1e-14))
        U = U[:, 1:χ]
        S = S[1:χ]
        V = V[:, 1:χ]
        mps.tensors[i] = reshape(U, (size(mps.tensors[i], 1), mps.d, χ))
        mps.tensors[i+1] = reshape(Diagonal(S) * V' * 
                                   reshape(mps.tensors[i+1], (size(mps.tensors[i+1], 1), :)), 
                                   (χ, mps.d, size(mps.tensors[i+1], 3)))
    end
    mps.χ = maximum(size.(mps.tensors, 3))
end

# Compute the expectation value of a local operator
function expect_local(mps::MPS, op::Matrix{ComplexF64}, site::Int)
    @assert size(op) == (mps.d, mps.d)
    E = ones(ComplexF64, 1, 1)
    for i in 1:mps.N
        if i == site
            E = E * reshape(conj.(mps.tensors[i]), (size(mps.tensors[i], 1), :)) *
                kron(I(size(mps.tensors[i], 3)), op) *
                reshape(mps.tensors[i], (size(mps.tensors[i], 1), :))'
        else
            E = E * reshape(conj.(mps.tensors[i]), (size(mps.tensors[i], 1), :)) *
                reshape(mps.tensors[i], (size(mps.tensors[i], 1), :))'
        end
    end
    return real(E[1, 1])
end