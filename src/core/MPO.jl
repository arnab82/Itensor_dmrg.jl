"""A dense MPO with tensors `(left, physical_out, physical_in, right)`."""
struct MPO
    tensors::Vector{Array{ComplexF64,4}}
    N::Int
    d::Int

    function MPO(tensors::Vector{Array{ComplexF64,4}})
        isempty(tensors) && throw(ArgumentError("an MPO needs at least one site"))
        d = size(first(tensors), 2)
        size(first(tensors), 1) == 1 || throw(DimensionMismatch("left boundary bond must be 1"))
        size(last(tensors), 4) == 1 || throw(DimensionMismatch("right boundary bond must be 1"))
        all(size(W, 2) == d && size(W, 3) == d for W in tensors) ||
            throw(DimensionMismatch("MPO physical dimensions differ"))
        all(size(tensors[i], 4) == size(tensors[i + 1], 1) for i in 1:length(tensors)-1) ||
            throw(DimensionMismatch("neighboring MPO bonds differ"))
        new(tensors, length(tensors), d)
    end
end

MPO(tensors::Vector{<:AbstractArray{<:Number,4}}) =
    MPO([Array{ComplexF64,4}(W) for W in tensors])

# Compatibility with the original field name.
function Base.getproperty(H::MPO, name::Symbol)
    name === :tensor && return getfield(H, :tensors)
    name === :d1 && return getfield(H, :d)
    name === :d2 && return getfield(H, :d)
    return getfield(H, name)
end

"""Return the full operator matrix. Intended for tests and small-system comparisons."""
function dense(H::MPO)
    block = H.tensors[1]
    for i in 2:H.N
        W = H.tensors[i]
        @tensor joined[l, o1, i1, o2, i2, r] := block[l, o1, i1, m] * W[m, o2, i2, r]
        ordered = permutedims(joined, (1, 2, 4, 3, 5, 6))
        block = reshape(ordered, size(joined, 1), size(joined, 2) * size(joined, 4),
                        size(joined, 3) * size(joined, 5), size(joined, 6))
    end
    return reshape(block, H.d^H.N, H.d^H.N)
end

MPO_to_array(H::MPO) = dense(H)
