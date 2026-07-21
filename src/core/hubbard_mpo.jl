# One-dimensional Fermi-Hubbard chain as a nearest-neighbor MPO.
#
# Each site is a spinful electron orbital with local dimension d = 4 and basis
# order matching ITensor's "Electron" site type:
#
#   1: |0⟩   (empty)
#   2: |↑⟩   (spin up)
#   3: |↓⟩   (spin down)
#   4: |↑↓⟩  (doubly occupied)
#
# The model is
#
#   H = -t Σ_{⟨i,i+1⟩,σ} (c†_{iσ} c_{i+1,σ} + h.c.)
#       + U Σ_i n_{i↑} n_{i↓}
#       - μ Σ_i (n_{i↑} + n_{i↓}).
#
# Fermions are handled by a site-level Jordan-Wigner string: a fermionic
# operator at site j carries ∏_{k<j} F_k, where F = diag(1,-1,-1,1) is the
# on-site fermion parity (-1)^(n↑+n↓). For NEAREST-neighbor hopping the string
# collapses to a single on-site F, so each hopping term is a genuine two-site
# operator L_i R_{i+1} and fits `nearest_neighbor_mpo`:
#
#   c†_{iσ} c_{i+1,σ}       = (Cdagσ · F)_i ⊗ (Cσ)_{i+1}
#   c†_{i+1,σ} c_{iσ} (h.c.)= (F · Cσ)_i   ⊗ (Cdagσ)_{i+1}
#
# The intra-site up/dn ordering signs are already baked into Cdn/Cdagdn (the
# −1 entries), exactly as in ITensor's Electron operators. The chain is uniform,
# so the translation-invariant builder applies directly.
#
# No quantum-number symmetry is used: this is a dense d=4 DMRG, so `dmrg` finds
# the GLOBAL (all particle-number sectors) ground state. Set `mu = U/2` for the
# particle-hole-symmetric point, where half filling is the global ground state.

"""
    electron_operators(T=ComplexF64) -> NamedTuple

Return the `4×4` single-site electron operators in the `|0⟩,|↑⟩,|↓⟩,|↑↓⟩` basis
(ITensor "Electron" convention): `Id, Cup, Cdagup, Cdn, Cdagdn, Nup, Ndn,
Nupdn, F`. `F = diag(1,-1,-1,1)` is the on-site fermion parity used for the
Jordan-Wigner string.
"""
function electron_operators(::Type{T}=ComplexF64) where {T<:Number}
    Id = Matrix{T}(I, 4, 4)

    Cup = zeros(T, 4, 4)
    Cup[1, 2] = 1          # |↑⟩  -> |0⟩
    Cup[3, 4] = 1          # |↑↓⟩ -> |↓⟩
    Cdagup = permutedims(Cup)  # real transpose (adjoint)

    Cdn = zeros(T, 4, 4)
    Cdn[1, 3] = 1          # |↓⟩  -> |0⟩
    Cdn[2, 4] = -1         # |↑↓⟩ -> -|↑⟩   (anticommute past the up electron)
    Cdagdn = permutedims(Cdn)

    Nup = zeros(T, 4, 4);  Nup[2, 2] = 1; Nup[4, 4] = 1
    Ndn = zeros(T, 4, 4);  Ndn[3, 3] = 1; Ndn[4, 4] = 1
    Nupdn = zeros(T, 4, 4); Nupdn[4, 4] = 1
    F = Matrix{T}(Diagonal(T[1, -1, -1, 1]))

    return (; Id, Cup, Cdagup, Cdn, Cdagdn, Nup, Ndn, Nupdn, F)
end

"""
    hubbard_mpo(N; t=1.0, U=4.0, mu=0.0, T=ComplexF64) -> MPO

Build the MPO for the open 1D Fermi-Hubbard chain of `N` electron sites
(local dimension 4) with hopping `t`, on-site repulsion `U`, and chemical
potential `mu`:

```
H = -t Σ_{⟨i,i+1⟩,σ} (c†_{iσ} c_{i+1,σ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
    - μ Σ_i (n_{i↑} + n_{i↓}).
```

The solver carries no particle-number symmetry, so `dmrg` returns the global
ground state over all fillings; use `mu = U/2` to place half filling at the
global minimum. The model is real — pass `T=Float64` for a fully real solve.
"""
function hubbard_mpo(N::Integer; t::Real=1.0, U::Real=4.0, mu::Real=0.0,
                     T::Type{<:Number}=ComplexF64)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    op = electron_operators(T)

    onsite = [(U, op.Nupdn)]
    if mu != 0
        push!(onsite, (-mu, op.Nup))
        push!(onsite, (-mu, op.Ndn))
    end

    bond = [
        (-t, op.Cdagup * op.F, op.Cup),     # c†_{i↑} c_{i+1,↑}
        (-t, op.F * op.Cup,   op.Cdagup),   # h.c.
        (-t, op.Cdagdn * op.F, op.Cdn),     # c†_{i↓} c_{i+1,↓}
        (-t, op.F * op.Cdn,   op.Cdagdn),   # h.c.
    ]

    return nearest_neighbor_mpo(N, 4; onsite=onsite, bond=bond, T=T)
end
