"""
    MPS{T}

A dense, open-boundary matrix-product state with tensors ordered
`(left, physical, right)` and scalar type `T` (e.g. `ComplexF64` or `Float64`).
Construct one from a vector of rank-3 arrays; mixed element types are promoted to
a common `T`.
"""
mutable struct MPS{T<:Number}
    tensors::Vector{Array{T,3}}
    N::Int
    d::Int

    function MPS(tensors::Vector{Array{T,3}}) where {T<:Number}
        isempty(tensors) && throw(ArgumentError("an MPS needs at least one site"))
        d = size(first(tensors), 2)
        size(first(tensors), 1) == 1 || throw(DimensionMismatch("left boundary bond must be 1"))
        size(last(tensors), 3) == 1 || throw(DimensionMismatch("right boundary bond must be 1"))
        all(size(A, 2) == d for A in tensors) || throw(DimensionMismatch("physical dimensions differ"))
        all(size(tensors[i], 3) == size(tensors[i + 1], 1) for i in 1:length(tensors)-1) ||
            throw(DimensionMismatch("neighboring MPS bonds differ"))
        new{T}(tensors, length(tensors), d)
    end
end

function MPS(tensors::AbstractVector{<:AbstractArray{<:Number,3}})
    T = isempty(tensors) ? ComplexF64 : promote_type(map(eltype, tensors)...)
    return MPS([Array{T,3}(A) for A in tensors])
end

Base.eltype(::MPS{T}) where {T} = T
Base.eltype(::Type{MPS{T}}) where {T} = T

"""Copy `psi`, converting its tensors to scalar type `T` (a no-op if already `T`)."""
_promote_mps(psi::MPS, ::Type{T}) where {T} =
    eltype(psi) == T ? copy(psi) : MPS([Array{T,3}(A) for A in psi.tensors])

Base.copy(psi::MPS) = MPS(deepcopy(psi.tensors))
bond_dimensions(psi::MPS) = [size(psi.tensors[i], 3) for i in 1:psi.N-1]

"""
    random_MPS(N, d, maxdim; rng=default_rng(), T=ComplexF64)

Create a normalized random `MPS{T}` without impossible oversized edge bonds.
Pass `T=Float64` for a real state.
"""
function random_MPS(N::Integer, d::Integer, maxdim::Integer;
                    rng=Random.default_rng(), T::Type{<:Number}=ComplexF64)
    N >= 2 || throw(ArgumentError("two-site DMRG requires N >= 2"))
    d >= 1 || throw(ArgumentError("d must be positive"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    bonds = [1; [min(maxdim, d^min(i, N - i)) for i in 1:N-1]; 1]
    tensors = [randn(rng, T, bonds[i], d, bonds[i + 1]) for i in 1:N]
    psi = MPS(tensors)
    right_canonicalize!(psi)
    normalize!(psi)
    return psi
end

function left_canonicalize!(psi::MPS)
    for i in 1:psi.N-1
        A = psi.tensors[i]
        l, d, r = size(A)
        F = qr(reshape(A, l * d, r))
        q = min(l * d, r)
        Q = Matrix(F.Q)[:, 1:q]
        R = Matrix(F.R)[1:q, :]
        psi.tensors[i] = reshape(Q, l, d, q)
        B = psi.tensors[i + 1]
        psi.tensors[i + 1] = reshape(R * reshape(B, r, :), q, d, size(B, 3))
    end
    return psi
end

function right_canonicalize!(psi::MPS)
    for i in psi.N:-1:2
        A = psi.tensors[i]
        l, d, r = size(A)
        F = qr(adjoint(reshape(A, l, d * r)))
        q = min(l, d * r)
        Q = Matrix(F.Q)[:, 1:q]
        R = Matrix(F.R)[1:q, :]
        psi.tensors[i] = reshape(adjoint(Q), q, d, r)
        B = psi.tensors[i - 1]
        psi.tensors[i - 1] = reshape(reshape(B, :, l) * adjoint(R), size(B, 1), d, q)
    end
    return psi
end

# Backward-compatible spellings.
left_normalize!(psi::MPS) = left_canonicalize!(psi)
right_normalize!(psi::MPS) = right_canonicalize!(psi)

function overlap(bra::MPS, ket::MPS)
    (bra.N, bra.d) == (ket.N, ket.d) || throw(DimensionMismatch("incompatible MPSs"))
    env = ones(promote_type(eltype(bra), eltype(ket)), 1, 1)
    for i in 1:bra.N
        A, B = bra.tensors[i], ket.tensors[i]
        @tensor next_env[ra, rb] := conj(A[la, s, ra]) * env[la, lb] * B[lb, s, rb]
        env = next_env
    end
    return env[]
end

LinearAlgebra.norm(psi::MPS) = sqrt(max(0.0, real(overlap(psi, psi))))

function LinearAlgebra.normalize!(psi::MPS)
    nrm = norm(psi)
    nrm > eps(Float64) || throw(ArgumentError("cannot normalize a zero MPS"))
    psi.tensors[1] ./= nrm
    return psi
end

"""Return the full state vector. Intended for tests and small-system comparisons."""
function dense(psi::MPS)
    state = psi.tensors[1]
    for i in 2:psi.N
        A = psi.tensors[i]
        @tensor joined[l, s1, s2, r] := state[l, s1, m] * A[m, s2, r]
        state = reshape(joined, size(joined, 1), size(joined, 2) * size(joined, 3), size(joined, 4))
    end
    return vec(state)
end
