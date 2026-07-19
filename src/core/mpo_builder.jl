# Generic nearest-neighbor MPO builder.
#
# Compiles a translation-invariant 1D Hamiltonian made of on-site terms and
# nearest-neighbor two-site terms into an MPO, using the finite-state-machine
# (automaton) construction that generalizes `heisenberg_mpo`. With K distinct
# bond terms the MPO bond dimension is w = K + 2:
#
#   state w   = "no term started yet" (start / identity-so-far),
#   states 2..K+1 = "bond term k half-applied" (one per nearest-neighbor channel),
#   state 1   = "term completed" (identity ever after).
#
# The nonzero operator-valued entries of the bulk tensor W[a, s', s, b] are
#   W[w, w] = 𝟙,  W[1, 1] = 𝟙,                 (propagate identity)
#   W[w, 1] = Σ (on-site terms),               (start → done)
#   W[w, 1+k] = cₖ · Lₖ,  W[1+k, 1] = Rₖ,       (start → channel k → done)
# so a start→channel-k→done path lays down cₖ · (Lₖ)ᵢ (Rₖ)ᵢ₊₁ on a bond and a
# start→done path lays down an on-site term.

_as_op(d, op) =
    size(op) == (d, d) ? Array{ComplexF64}(op) :
        throw(DimensionMismatch("operator must be $(d)×$(d) to match the physical dimension"))

"""
    nearest_neighbor_mpo(N, d; onsite=[], bond=[]) -> MPO

Build the MPO for a length-`N`, physical-dimension-`d` open chain with the
translation-invariant Hamiltonian

```
H = Σᵢ Σ_onsite c · Ôᵢ  +  Σᵢ Σ_bond c · L̂ᵢ R̂ᵢ₊₁ .
```

`onsite` is a collection of `(coeff, op)` pairs and `bond` a collection of
`(coeff, opL, opR)` triples, with each `op` a `d×d` matrix. The resulting MPO
has bond dimension `length(bond) + 2`.

# Example
```julia
# spin-1/2 Heisenberg with field h_z, rebuilt from local terms
Sz = ComplexF64[0.5 0; 0 -0.5]; Sp = ComplexF64[0 1; 0 0]; Sm = ComplexF64[0 0; 1 0]
H = nearest_neighbor_mpo(N, 2;
        onsite = [(hz, Sz)],
        bond   = [(J/2, Sp, Sm), (J/2, Sm, Sp), (J, Sz, Sz)])
```
"""
function nearest_neighbor_mpo(N::Integer, d::Integer; onsite=(), bond=())
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    d >= 1 || throw(ArgumentError("d must be positive"))
    K = length(bond)
    w = K + 2
    id = Matrix{ComplexF64}(I, d, d)

    # Accumulate all on-site terms into a single d×d operator.
    O = zeros(ComplexF64, d, d)
    for (c, op) in onsite
        O .+= ComplexF64(c) .* _as_op(d, op)
    end

    bulk = zeros(ComplexF64, w, d, d, w)
    bulk[1, :, :, 1] .= id          # completed → completed
    bulk[w, :, :, w] .= id          # not-started → not-started
    bulk[w, :, :, 1] .= O           # on-site: start → done
    for (k, term) in enumerate(bond)
        c, L, R = term
        bulk[w, :, :, 1 + k] .= ComplexF64(c) .* _as_op(d, L)   # start → channel k
        bulk[1 + k, :, :, 1] .= _as_op(d, R)                    # channel k → done
    end

    tensors = Vector{Array{ComplexF64,4}}(undef, N)
    tensors[1] = copy(bulk[w:w, :, :, :])       # left boundary starts in state w
    for i in 2:N-1
        tensors[i] = copy(bulk)
    end
    tensors[N] = copy(bulk[:, :, :, 1:1])       # right boundary ends in state 1
    return MPO(tensors)
end

"""
    tfim_mpo(N; J=1.0, h=1.0) -> MPO

Transverse-field Ising model on an open spin-1/2 chain, in spin operators
(`Sᵅ = σᵅ/2`):

```math
H = -J Σᵢ Sᶻᵢ Sᶻᵢ₊₁ - h Σᵢ Sˣᵢ .
```

A convenience wrapper around [`nearest_neighbor_mpo`](@ref).
"""
function tfim_mpo(N::Integer; J::Real=1.0, h::Real=1.0)
    Sx = ComplexF64[0 0.5; 0.5 0]
    Sz = ComplexF64[0.5 0; 0 -0.5]
    return nearest_neighbor_mpo(N, 2; onsite=[(-h, Sx)], bond=[(-J, Sz, Sz)])
end
