# Abelian U(1) quantum numbers for the block-sparse ("symmetry=true") code path.
#
# A `QN{K}` bundles `K` independent additive U(1) charges as an `NTuple{K,Int}`.
# It is an isbits, immutable value, so it hashes and compares by content and can
# key the block dictionaries of a `SymTensor`. Examples:
#   * electron site conserves (N↑, N↓)  → K = 2
#   * spin-1/2 site conserves 2·Sz      → K = 1
#
# Charges are additive, and an index "arrow" (±1) flips the sign a leg's charge
# contributes to a tensor's conserved flux, so `QN` supports `+`, `-`, unary `-`,
# integer scaling, and a `zero`.

struct QN{K}
    q::NTuple{K,Int}
end

QN(qs::Integer...) = QN(NTuple{length(qs),Int}(Int.(qs)))

charges(a::QN) = a.q
Base.length(::QN{K}) where {K} = K

Base.:+(a::QN{K}, b::QN{K}) where {K} = QN{K}(a.q .+ b.q)
Base.:-(a::QN{K}, b::QN{K}) where {K} = QN{K}(a.q .- b.q)
Base.:-(a::QN{K}) where {K} = QN{K}(.-a.q)
Base.:*(s::Integer, a::QN{K}) where {K} = QN{K}(s .* a.q)
Base.:*(a::QN{K}, s::Integer) where {K} = s * a

Base.zero(::Type{QN{K}}) where {K} = QN{K}(ntuple(_ -> 0, K))
Base.zero(::QN{K}) where {K} = zero(QN{K})

# Deterministic ordering so sector lists and block iteration are reproducible.
Base.isless(a::QN{K}, b::QN{K}) where {K} = isless(a.q, b.q)

Base.show(io::IO, a::QN) = print(io, "QN", a.q)
