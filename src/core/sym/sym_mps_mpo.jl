# Symmetric (block-sparse) MPS and MPO built on `SymTensor`, for the
# `symmetry=true` path. The dense builders are reused verbatim: a charge-
# conserving dense MPO is turned into a `SymMPO` by *inferring* every bond
# automaton state's quantum number via flux propagation, so no builder needs to
# know about symmetry. A `SymMPS` is a random state pinned to one charge sector.
#
# Arrow / flux conventions (all tensors carry flux 0; the state's total charge
# lives on the right boundary bond). Bonds "point left" — a tensor's LEFT bond is
# outgoing (+1) and its RIGHT bond incoming (-1) — so that the left factor of a
# block-sparse QR/SVD (which appends an incoming bond) drops straight into place
# as a canonicalized MPS tensor:
#   ket MPS  A[l, s, r]           arrows (l:+1, s:+1, r:-1),  q_r = q_l + q_s
#   MPO      W[l, s', s, r]        arrows (l:+1, s':+1, s:-1, r:-1)
# so the ket physical leg (+1) is dual to the MPO physical-in leg (-1), and
# neighbouring bonds are duals. These are enforced automatically: `SymTensor`
# rejects any tensor whose weight lands outside the flux-0 blocks.

# ---------------------------------------------------------------------------
# Site quantum-number tables (basis order matches the dense operators)
# ---------------------------------------------------------------------------

"""Electron site charges (N↑, N↓) for basis `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`."""
electron_site_qns() = [QN(0, 0), QN(1, 0), QN(0, 1), QN(1, 1)]

"""Spin-1/2 site charge 2·Sz for basis `|↑⟩, |↓⟩`."""
spinhalf_site_qns() = [QN(1), QN(-1)]

"""Target sector for the half-filled electron chain: `N/2` up and `N/2` down."""
electron_half_filling(N::Integer) = QN(N ÷ 2, N ÷ 2)

# ---------------------------------------------------------------------------
# Operator charge and per-slot index construction
# ---------------------------------------------------------------------------

"""
    op_charge(op, site_qns) -> QN

Charge shift `Δ` of a local operator: `op[s',s] ≠ 0 ⇒ qn(s') - qn(s) = Δ`, the
same for every nonzero entry (the operator must be charge-homogeneous). The zero
operator maps to `zero`.
"""
function op_charge(op::AbstractMatrix, site_qns::Vector{QN{K}}; atol::Real=1e-12) where {K}
    Δ = nothing
    for s in axes(op, 2), sp in axes(op, 1)
        abs(op[sp, s]) > atol || continue
        d = site_qns[sp] - site_qns[s]
        if Δ === nothing
            Δ = d
        elseif d != Δ
            throw(ArgumentError("operator is not charge-homogeneous ($(Δ) vs $(d))"))
        end
    end
    return Δ === nothing ? zero(QN{K}) : Δ
end

# Build a SymIndex from per-slot QN labels, plus the permutation that groups the
# slots by sorted sector (so a dense axis can be reordered to the block layout).
function _index_from_labels(arrow::Integer, labels::Vector{QN{K}}) where {K}
    uq = sort(unique(labels))
    perm = Int[]
    sizes = Int[]
    for q in uq
        idxs = findall(==(q), labels)
        append!(perm, idxs)
        push!(sizes, length(idxs))
    end
    return SymIndex(arrow, uq, sizes), perm
end

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""Block-sparse MPO: rank-4 `SymTensor`s plus the basis-order site charges."""
struct SymMPO{T,K}
    tensors::Vector{SymTensor{T,4,K}}
    sites::Vector{Vector{QN{K}}}
    N::Int
    d::Int
end

"""Block-sparse MPS pinned to total charge `sector`."""
struct SymMPS{T,K}
    tensors::Vector{SymTensor{T,3,K}}
    sites::Vector{Vector{QN{K}}}
    sector::QN{K}
    N::Int
    d::Int
end

Base.eltype(::SymMPO{T}) where {T} = T
Base.eltype(::SymMPS{T}) where {T} = T
Base.length(H::SymMPO) = H.N
Base.length(psi::SymMPS) = psi.N

# ---------------------------------------------------------------------------
# Symmetrizing a dense MPO
# ---------------------------------------------------------------------------

"""
    symmetrize_mpo(H::MPO, sites) -> SymMPO

Turn a charge-conserving dense `MPO` into a block-sparse `SymMPO`, given the
per-site basis-order charge tables `sites`. Bond quantum numbers are inferred by
propagating flux from the (trivial) left boundary; a non-conserving MPO raises an
error.
"""
function symmetrize_mpo(H::MPO{T}, sites::Vector{Vector{QN{K}}}; atol::Real=1e-10) where {T,K}
    N = H.N
    length(sites) == N || throw(ArgumentError("need one site table per MPO site"))
    bdim(b) = b <= N ? size(H.tensors[b], 1) : size(H.tensors[N], 4)
    # Each bond automaton state gets a QN. A zero on-site term can leave a state
    # unreachable in one direction, so propagate BOTH ways (φ_right = φ_left + Δ)
    # to a fixed point; every state is then pinned via some nonzero path.
    part = [Vector{Union{Nothing,QN{K}}}(nothing, bdim(b)) for b in 1:N + 1]
    part[1][1] = zero(QN{K})                        # left boundary: trivial
    changed = true
    while changed
        changed = false
        for n in 1:N
            W = H.tensors[n]
            Dl, d1, d2, Dr = size(W)
            (Dl == length(part[n]) && d1 == d2 == length(sites[n])) ||
                throw(DimensionMismatch("MPO tensor $n has an unexpected shape"))
            sq = sites[n]
            for a in 1:Dl, b in 1:Dr, sp in 1:d1, s in 1:d2
                abs(W[a, sp, s, b]) > atol || continue
                Δ = sq[sp] - sq[s]
                L, R = part[n][a], part[n + 1][b]
                if L !== nothing && R === nothing
                    part[n + 1][b] = L + Δ; changed = true
                elseif L === nothing && R !== nothing
                    part[n][a] = R - Δ; changed = true
                elseif L !== nothing && R !== nothing && L + Δ != R
                    throw(ArgumentError("MPO does not conserve the supplied charges (site $n)"))
                end
            end
        end
    end
    for b in 1:N + 1
        any(isnothing, part[b]) &&
            throw(ArgumentError("MPO bond $b has an unreachable state (disconnected automaton)"))
    end
    bondphi = [Vector{QN{K}}(part[b]) for b in 1:N + 1]

    tensors = Vector{SymTensor{T,4,K}}(undef, N)
    for n in 1:N
        W = H.tensors[n]
        leftidx, pl = _index_from_labels(+1, bondphi[n])
        rightidx, pr = _index_from_labels(-1, bondphi[n + 1])
        outidx, pp = _index_from_labels(+1, sites[n])          # physical-out (bra)
        inidx = SymIndex(-1, copy(outidx.sectors), copy(outidx.sizes))  # physical-in (ket)
        Wp = W[pl, pp, pp, pr]
        tensors[n] = SymTensor(Wp, (leftidx, outidx, inidx, rightidx), zero(QN{K}); atol=atol)
    end
    return SymMPO{T,K}(tensors, sites, N, H.d)
end

"""Reconstruct a dense `MPO` (physical axes restored to basis order)."""
function dense(H::SymMPO{T,K}) where {T,K}
    tensors = Vector{Array{T,4}}(undef, H.N)
    for n in 1:H.N
        Wd = dense(H.tensors[n])
        _, pp = _index_from_labels(+1, H.sites[n])
        inv = invperm(pp)
        tensors[n] = Wd[:, inv, inv, :]                        # bonds stay sorted (consistent)
    end
    return MPO(tensors)
end

# ---------------------------------------------------------------------------
# Random symmetric MPS in a fixed sector
# ---------------------------------------------------------------------------

"""
    random_sym_mps(sites, sector, maxdim; T=ComplexF64, rng, perbond=1) -> SymMPS

A random block-sparse MPS whose total charge is `sector`. Bond charges are the
charges reachable both forward from the trivial left boundary and backward from
`sector`; each carries `perbond` states (boundaries are dimension 1).
"""
function random_sym_mps(sites::Vector{Vector{QN{K}}}, sector::QN{K}, maxdim::Integer;
                        T::Type{<:Number}=ComplexF64, rng=Random.default_rng(),
                        perbond::Integer=1) where {K}
    N = length(sites)
    fwd = Vector{Set{QN{K}}}(undef, N + 1)
    fwd[1] = Set([zero(QN{K})])
    for n in 1:N
        fwd[n + 1] = Set(q + sq for q in fwd[n] for sq in sites[n])
    end
    bwd = Vector{Set{QN{K}}}(undef, N + 1)
    bwd[N + 1] = Set([sector])
    for n in N:-1:1
        bwd[n] = Set(q - sq for q in bwd[n + 1] for sq in sites[n])
    end
    keepq = [sort(collect(intersect(fwd[b], bwd[b]))) for b in 1:N + 1]
    isempty(keepq[N + 1]) && throw(ArgumentError("sector $sector is unreachable"))

    # bond dimension per charge (boundaries forced to a single trivial state)
    dims = [Dict{QN{K},Int}() for _ in 1:N + 1]
    dims[1][zero(QN{K})] = 1
    dims[N + 1][sector] = 1
    for b in 2:N, q in keepq[b]
        dims[b][q] = min(Int(maxdim), Int(perbond))
    end

    bondindex(b, arrow) = SymIndex(arrow, copy(keepq[b]), [dims[b][q] for q in keepq[b]])
    tensors = Vector{SymTensor{T,3,K}}(undef, N)
    for n in 1:N
        leftidx = bondindex(n, +1)
        rightidx = bondindex(n + 1, -1)
        physidx, _ = _index_from_labels(+1, sites[n])
        legs = (leftidx, physidx, rightidx)
        blocks = Dict{NTuple{3,Int},Array{T,3}}()
        for key in _allowed_keys(legs, zero(QN{K}))
            blocks[key] = randn(rng, T, _blockdims(legs, key))
        end
        isempty(blocks) && throw(ArgumentError("empty MPS tensor at site $n"))
        tensors[n] = SymTensor{T,3,K}(legs, zero(QN{K}), blocks)
    end
    return SymMPS{T,K}(tensors, sites, sector, N, length(sites[1]))
end

"""Reconstruct a dense `MPS` (physical axis restored to basis order)."""
function dense(psi::SymMPS{T,K}) where {T,K}
    tensors = Vector{Array{T,3}}(undef, psi.N)
    for n in 1:psi.N
        Ad = dense(psi.tensors[n])
        _, pp = _index_from_labels(+1, psi.sites[n])
        tensors[n] = Ad[:, invperm(pp), :]
    end
    return MPS(tensors)
end

bond_dimensions(psi::SymMPS) = [totaldim(psi.tensors[i].legs[3]) for i in 1:psi.N - 1]

"""
    random_MPS(H::SymMPO, maxdim; sector, rng, T, perbond=1) -> SymMPS

Ergonomic symmetric initial state: read the site charges from a `SymMPO` and
build a random MPS in `sector`.
"""
function random_MPS(H::SymMPO, maxdim::Integer; sector::QN,
                    rng=Random.default_rng(), T::Type{<:Number}=eltype(H), perbond::Integer=1)
    return random_sym_mps(H.sites, sector, maxdim; T=T, rng=rng, perbond=perbond)
end

# ---------------------------------------------------------------------------
# Observables (measured on the dense state — fine for measurement-sized systems)
# ---------------------------------------------------------------------------

expect(psi::SymMPS, op::AbstractMatrix, site::Integer) = expect(dense(psi), op, site)
expect(psi::SymMPS, op::AbstractMatrix) = expect(dense(psi), op)
correlation(psi::SymMPS, op1::AbstractMatrix, op2::AbstractMatrix, i::Integer, j::Integer) =
    correlation(dense(psi), op1, op2, i, j)
