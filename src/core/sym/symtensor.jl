# Block-sparse tensors for the Abelian-U(1) ("symmetry=true") code path.
#
# A `SymTensor` is a dense-block-sparse tensor: each leg is a `SymIndex` carrying
# an arrow (±1) and a partition of its dimension into charge sectors, and only the
# blocks whose signed charge sum equals the tensor's conserved `flux` are stored.
# All heavy operations (contraction, SVD, QR) run block-by-block, which is exactly
# where the memory/time win over the dense path comes from: with `s` charge
# sectors a leg's blocks are ~`1/s` the size of the full index.
#
# Conventions:
#   * arrow +1 = outgoing, -1 = incoming;
#   * a block keyed by sector-indices `key` is allowed iff
#       Σ_leg  arrow_leg · sectors_leg[key_leg]  ==  flux.
#
# This file provides the standalone tensor algebra; the MPS/MPO wrappers and the
# symmetric DMRG engine build on top of it.

# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------

"""
    SymIndex{K}(arrow, sectors, sizes)

One leg of a `SymTensor`: an `arrow ∈ {+1,-1}` and a list of charge `sectors`
(`QN{K}`) with matching block `sizes`. Sectors are stored unique and sorted, so a
sector's position is a stable "sector index".
"""
struct SymIndex{K}
    arrow::Int
    sectors::Vector{QN{K}}
    sizes::Vector{Int}

    function SymIndex(arrow::Integer, sectors::Vector{QN{K}}, sizes::Vector{<:Integer}) where {K}
        (arrow == 1 || arrow == -1) || throw(ArgumentError("arrow must be ±1"))
        length(sectors) == length(sizes) ||
            throw(DimensionMismatch("sectors and sizes must have equal length"))
        allunique(sectors) || throw(ArgumentError("sectors must be unique"))
        all(>(0), sizes) || throw(ArgumentError("sector sizes must be positive"))
        p = sortperm(sectors)
        new{K}(Int(arrow), sectors[p], Int.(sizes)[p])
    end
end

nsectors(ix::SymIndex) = length(ix.sectors)
totaldim(ix::SymIndex) = sum(ix.sizes; init=0)
dual(ix::SymIndex) = SymIndex(-ix.arrow, copy(ix.sectors), copy(ix.sizes))
sectorindex(ix::SymIndex, q::QN) = findfirst(==(q), ix.sectors)

# Cumulative offsets: sector s occupies rows/cols `offsets[s]+1 : offsets[s+1]`.
function _offsets(ix::SymIndex)
    offs = Vector{Int}(undef, nsectors(ix) + 1)
    offs[1] = 0
    for s in 1:nsectors(ix)
        offs[s + 1] = offs[s] + ix.sizes[s]
    end
    return offs
end

function Base.:(==)(a::SymIndex, b::SymIndex)
    a.arrow == b.arrow && a.sectors == b.sectors && a.sizes == b.sizes
end

# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

"""
    SymTensor{T,R,K}

Rank-`R` block-sparse tensor over scalar `T` with `K` conserved U(1) charges.
`legs` are the `R` `SymIndex`es, `flux` the conserved total charge, and `blocks`
maps a per-leg sector-index tuple to its dense block.
"""
struct SymTensor{T,R,K}
    legs::NTuple{R,SymIndex{K}}
    flux::QN{K}
    blocks::Dict{NTuple{R,Int},Array{T,R}}
end

Base.eltype(::SymTensor{T}) where {T} = T
Base.eltype(::Type{<:SymTensor{T}}) where {T} = T
symrank(::SymTensor{T,R}) where {T,R} = R
numblocks(A::SymTensor) = length(A.blocks)

# Signed charge sum of a block key.
function _block_flux(legs::NTuple{R,SymIndex{K}}, key) where {R,K}
    f = zero(QN{K})
    for l in 1:R
        f += legs[l].arrow * legs[l].sectors[key[l]]
    end
    return f
end

# All sector-index tuples whose block is allowed by flux conservation.
function _allowed_keys(legs::NTuple{R,SymIndex{K}}, flux::QN{K}) where {R,K}
    axes_ = ntuple(l -> 1:nsectors(legs[l]), R)
    keys = NTuple{R,Int}[]
    for key in Iterators.product(axes_...)
        _block_flux(legs, key) == flux && push!(keys, key)
    end
    return keys
end

function _blockdims(legs::NTuple{R,SymIndex{K}}, key) where {R,K}
    ntuple(l -> legs[l].sizes[key[l]], R)
end

"""
    SymTensor(dense, legs, flux; atol=1e-10)

Build a block-sparse tensor from a dense array by slicing out the flux-allowed
blocks. Entries outside those blocks must vanish (they are the symmetry-forbidden
amplitudes); this is checked to `atol`.
"""
function SymTensor(dense::AbstractArray{T,R}, legs::NTuple{R,SymIndex{K}},
                   flux::QN{K}; atol::Real=1e-8) where {T,R,K}
    size(dense) == ntuple(l -> totaldim(legs[l]), R) ||
        throw(DimensionMismatch("dense size does not match the leg dimensions"))
    offs = ntuple(l -> _offsets(legs[l]), R)
    ranges_of(key) = ntuple(l -> (offs[l][key[l]] + 1):(offs[l][key[l] + 1]), R)
    blocks = Dict{NTuple{R,Int},Array{T,R}}()
    for key in _allowed_keys(legs, flux)
        blocks[key] = Array{T,R}(dense[ranges_of(key)...])
    end
    # Forbidden weight, summed directly (subtracting two big near-equal sums would
    # bury real misplacements under summation roundoff).
    forbidden = 0.0
    for key in Iterators.product(ntuple(l -> 1:nsectors(legs[l]), R)...)
        _block_flux(legs, key) == flux && continue
        forbidden += sum(abs2, @view dense[ranges_of(key)...])
    end
    sqrt(forbidden) <= atol * max(1, sqrt(sum(abs2, dense))) ||
        throw(ArgumentError("dense array has weight $(sqrt(forbidden)) outside the flux-$flux blocks"))
    return SymTensor{T,R,K}(legs, flux, blocks)
end

"""Reconstruct the full dense array (for tests and small-system comparison)."""
function dense(A::SymTensor{T,R,K}) where {T,R,K}
    dims = ntuple(l -> totaldim(A.legs[l]), R)
    out = zeros(T, dims)
    offs = ntuple(l -> _offsets(A.legs[l]), R)
    for (key, blk) in A.blocks
        ranges = ntuple(l -> (offs[l][key[l]] + 1):(offs[l][key[l] + 1]), R)
        out[ranges...] = blk
    end
    return out
end

LinearAlgebra.norm(A::SymTensor) = sqrt(sum(b -> sum(abs2, b), values(A.blocks); init=0.0))

function Base.:*(alpha::Number, A::SymTensor{T,R,K}) where {T,R,K}
    S = promote_type(typeof(alpha), T)
    blocks = Dict{NTuple{R,Int},Array{S,R}}(k => S(alpha) .* v for (k, v) in A.blocks)
    return SymTensor{S,R,K}(A.legs, A.flux, blocks)
end
Base.:*(A::SymTensor, alpha::Number) = alpha * A

# ---------------------------------------------------------------------------
# Contraction
# ---------------------------------------------------------------------------

# Matricize a block with `kept` legs as rows and `contr` legs as columns.
function _mat(blk::AbstractArray, kept::Tuple, contr::Tuple)
    p = permutedims(blk, (kept..., contr...))
    kd = prod(i -> size(blk, i), kept; init=1)
    cd = prod(i -> size(blk, i), contr; init=1)
    return reshape(p, kd, cd)
end

"""
    contract(A, B, cA, cB) -> SymTensor

Contract legs `cA[i]` of `A` with legs `cB[i]` of `B` (equal length, matched
pairwise). Contracted legs must be duals: opposite arrows and identical sector
structure. The result carries the un-contracted legs of `A` followed by those of
`B`, and flux `A.flux + B.flux`.
"""
function contract(A::SymTensor{TA,RA,K}, B::SymTensor{TB,RB,K},
                  cA, cB) where {TA,TB,RA,RB,K}
    length(cA) == length(cB) || throw(ArgumentError("cA and cB must have equal length"))
    m = length(cA)
    for i in 1:m
        la, lb = A.legs[cA[i]], B.legs[cB[i]]
        la.arrow == -lb.arrow || throw(ArgumentError("contracted legs must have opposite arrows"))
        (la.sectors == lb.sectors && la.sizes == lb.sizes) ||
            throw(ArgumentError("contracted legs have incompatible sector structure"))
    end
    keptA = Tuple(setdiff(1:RA, cA))
    keptB = Tuple(setdiff(1:RB, cB))
    cAt, cBt = Tuple(cA), Tuple(cB)
    RC = length(keptA) + length(keptB)
    T = promote_type(TA, TB)
    outlegs = (ntuple(j -> A.legs[keptA[j]], length(keptA))...,
               ntuple(j -> B.legs[keptB[j]], length(keptB))...)::NTuple{RC,SymIndex{K}}

    # Index B blocks by the sector-indices on their contracted legs. Contracted
    # legs are duals with identical sector order, so a matching A block shares the
    # same contracted sector-indices.
    Bby = Dict{NTuple{m,Int},Vector{Tuple{NTuple{RB,Int},Array{TB,RB}}}}()
    for (bk, blk) in B.blocks
        sig = ntuple(i -> bk[cBt[i]], m)
        push!(get!(Bby, sig, Vector{Tuple{NTuple{RB,Int},Array{TB,RB}}}()), (bk, blk))
    end

    blocks = Dict{NTuple{RC,Int},Array{T,RC}}()
    for (ak, ablk) in A.blocks
        sig = ntuple(i -> ak[cAt[i]], m)
        matches = get(Bby, sig, nothing)
        matches === nothing && continue
        Amat = _mat(ablk, keptA, cAt)                     # (keptA, contr)
        for (bk, bblk) in matches
            Bmat = _mat(bblk, cBt, keptB)                 # (contr, keptB)
            Cmat = Amat * Bmat                            # (keptA, keptB)
            outkey = (ntuple(j -> ak[keptA[j]], length(keptA))...,
                      ntuple(j -> bk[keptB[j]], length(keptB))...)::NTuple{RC,Int}
            cdims = _blockdims(outlegs, outkey)
            contrib = reshape(Cmat, cdims)
            if haskey(blocks, outkey)
                blocks[outkey] .+= contrib
            else
                blocks[outkey] = Array{T,RC}(contrib)
            end
        end
    end
    return SymTensor{T,RC,K}(outlegs, A.flux + B.flux, blocks)
end

# ---------------------------------------------------------------------------
# Fusing legs into a single (block-diagonal) matrix axis
# ---------------------------------------------------------------------------

# For a subset of legs, enumerate every sector-index combination, group by the
# combination's signed charge, and lay the combinations out contiguously within
# each charge. Returns:
#   place : combo -> (charge, offset, size)   (for filling the matrix)
#   dims  : charge -> fused dimension
#   groups: charge -> sorted Vector{(combo, offset, size)}   (for rebuilding)
function _fuse(legs::NTuple{R,SymIndex{K}}, subset::Tuple) where {R,K}
    n = length(subset)
    place = Dict{NTuple{n,Int},Tuple{QN{K},Int,Int}}()
    groups = Dict{QN{K},Vector{Tuple{NTuple{n,Int},Int,Int}}}()
    axes_ = ntuple(j -> 1:nsectors(legs[subset[j]]), n)
    for combo in Iterators.product(axes_...)
        q = zero(QN{K})
        sz = 1
        for j in 1:n
            leg = legs[subset[j]]
            q += leg.arrow * leg.sectors[combo[j]]
            sz *= leg.sizes[combo[j]]
        end
        push!(get!(groups, q, Vector{Tuple{NTuple{n,Int},Int,Int}}()), (combo, 0, sz))
    end
    dims = Dict{QN{K},Int}()
    for (q, lst) in groups
        sort!(lst, by = x -> x[1])
        off = 0
        for i in eachindex(lst)
            combo, _, sz = lst[i]
            lst[i] = (combo, off, sz)
            place[combo] = (q, off, sz)
            off += sz
        end
        dims[q] = off
    end
    return place, dims, groups
end

# Assemble the flux-allowed blocks into per-charge dense matrices M[qr], where
# rows come from `row_legs` and columns from `col_legs` (block-diagonal because
# col charge is fixed to flux - qr).
function _matricize(A::SymTensor{T,R,K}, row_legs::Tuple, col_legs::Tuple) where {T,R,K}
    rplace, rdims, rgroups = _fuse(A.legs, row_legs)
    cplace, cdims, cgroups = _fuse(A.legs, col_legs)
    mats = Dict{QN{K},Matrix{T}}()
    for qr in keys(rdims)
        qc = A.flux - qr
        haskey(cdims, qc) || continue
        mats[qr] = zeros(T, rdims[qr], cdims[qc])
    end
    for (key, blk) in A.blocks
        rcombo = ntuple(j -> key[row_legs[j]], length(row_legs))
        ccombo = ntuple(j -> key[col_legs[j]], length(col_legs))
        qr, roff, rsz = rplace[rcombo]
        _, coff, csz = cplace[ccombo]
        M = _mat(blk, Tuple(row_legs), Tuple(col_legs))
        mats[qr][roff + 1:roff + rsz, coff + 1:coff + csz] = M
    end
    return mats, rgroups, cgroups
end

# Build the "left" factor (row_legs + new incoming bond, flux 0) from per-charge
# column matrices `cols[qr]` of shape (rowdim[qr], keep[qr]).
function _left_factor(cols::Dict{QN{K},Matrix{T}}, rgroups, row_legs::Tuple,
                      A::SymTensor{T,R,K}) where {T,R,K}
    bond_qs = sort!(collect(keys(cols)))
    bond = SymIndex(-1, copy(bond_qs), [size(cols[q], 2) for q in bond_qs])
    nrl = length(row_legs)
    legs = (ntuple(j -> A.legs[row_legs[j]], nrl)..., bond)::NTuple{nrl + 1,SymIndex{K}}
    blocks = Dict{NTuple{nrl + 1,Int},Array{T,nrl + 1}}()
    for (bidx, q) in enumerate(bond_qs)
        C = cols[q]                                   # (rowdim, keep)
        keep = size(C, 2)
        for (combo, off, sz) in rgroups[q]
            slice = C[off + 1:off + sz, :]            # (sz, keep)
            legsizes = ntuple(j -> A.legs[row_legs[j]].sizes[combo[j]], nrl)
            blk = reshape(slice, legsizes..., keep)
            blocks[(combo..., bidx)] = Array{T,nrl + 1}(blk)
        end
    end
    return SymTensor{T,nrl + 1,K}(legs, zero(QN{K}), blocks)
end

# Build the "right" factor (new outgoing bond + col_legs, flux = A.flux) from
# per-charge row matrices `rows[qr]` of shape (keep[qr], coldim[qc]).
function _right_factor(rows::Dict{QN{K},Matrix{T}}, cgroups, col_legs::Tuple,
                       A::SymTensor{T,R,K}) where {T,R,K}
    bond_qs = sort!(collect(keys(rows)))
    bond = SymIndex(+1, copy(bond_qs), [size(rows[q], 1) for q in bond_qs])
    ncl = length(col_legs)
    legs = (bond, ntuple(j -> A.legs[col_legs[j]], ncl)...)::NTuple{ncl + 1,SymIndex{K}}
    blocks = Dict{NTuple{ncl + 1,Int},Array{T,ncl + 1}}()
    for (bidx, qr) in enumerate(bond_qs)
        Rm = rows[qr]                                 # (keep, coldim)
        keep = size(Rm, 1)
        qc = A.flux - qr
        for (combo, off, sz) in cgroups[qc]
            slice = Rm[:, off + 1:off + sz]           # (keep, sz)
            legsizes = ntuple(j -> A.legs[col_legs[j]].sizes[combo[j]], ncl)
            blk = reshape(slice, keep, legsizes...)
            blocks[(bidx, combo...)] = Array{T,ncl + 1}(blk)
        end
    end
    return SymTensor{T,ncl + 1,K}(legs, A.flux, blocks)
end

# ---------------------------------------------------------------------------
# SVD and QR
# ---------------------------------------------------------------------------

"""
    svd_truncated(A, row_legs, col_legs; maxdim, cutoff) -> (U, S, V, discarded)

Block-sparse SVD across the bipartition (`row_legs` | `col_legs`). Each conserved
charge gives an independent dense SVD; singular values are truncated *globally*
across all charges to at most `maxdim` values, dropping the smallest whose summed
weight stays under `cutoff`. `U` (row legs + incoming bond, flux 0) and `V`
(outgoing bond + col legs, flux `A.flux`) are isometries; `S` maps each surviving
bond charge to its singular values. `discarded` is the dropped squared weight.
"""
function svd_truncated(A::SymTensor{T,R,K}, row_legs, col_legs;
                       maxdim::Integer, cutoff::Real) where {T,R,K}
    row_legs, col_legs = Tuple(row_legs), Tuple(col_legs)
    mats, rgroups, cgroups = _matricize(A, row_legs, col_legs)

    Us = Dict{QN{K},Matrix{T}}()
    Ss = Dict{QN{K},Vector{real(T)}}()
    Vts = Dict{QN{K},Matrix{T}}()
    labelled = Tuple{real(T),QN{K}}[]              # (singular value, charge)
    for (qr, M) in mats
        F = svd(M)
        Us[qr] = F.U
        Ss[qr] = F.S
        Vts[qr] = F.Vt
        for sv in F.S
            push!(labelled, (sv, qr))
        end
    end
    isempty(labelled) && throw(ArgumentError("SVD of an empty tensor"))

    allS = sort([s for (s, _) in labelled]; rev=true)
    keep = kept_dimension(allS, maxdim, cutoff)
    # Count how many singular values per charge lie in the kept set (break ties by
    # the global rank so the total is exactly `keep`).
    order = sortperm(labelled; by = x -> x[1], rev=true)
    keepq = Dict{QN{K},Int}()
    discarded = 0.0
    for (rank_i, idx) in enumerate(order)
        sv, q = labelled[idx]
        if rank_i <= keep
            keepq[q] = get(keepq, q, 0) + 1
        else
            discarded += abs2(sv)
        end
    end

    Uk = Dict{QN{K},Matrix{T}}()
    Sk = Dict{QN{K},Vector{real(T)}}()
    Vk = Dict{QN{K},Matrix{T}}()
    for (q, k) in keepq
        k == 0 && continue
        Uk[q] = Us[q][:, 1:k]
        Sk[q] = Ss[q][1:k]
        Vk[q] = Vts[q][1:k, :]
    end
    U = _left_factor(Uk, rgroups, row_legs, A)
    V = _right_factor(Vk, cgroups, col_legs, A)
    return U, Sk, V, discarded
end

"""
    qr_factorize(A, row_legs, col_legs) -> (Q, Rf)

Block-sparse (thin) QR across the bipartition. `Q` (row legs + incoming bond,
flux 0) is an isometry and `Rf` (outgoing bond + col legs, flux `A.flux`) is upper
triangular per charge. Used for MPS canonicalization.
"""
function qr_factorize(A::SymTensor{T,R,K}, row_legs, col_legs) where {T,R,K}
    row_legs, col_legs = Tuple(row_legs), Tuple(col_legs)
    mats, rgroups, cgroups = _matricize(A, row_legs, col_legs)
    Qs = Dict{QN{K},Matrix{T}}()
    Rs = Dict{QN{K},Matrix{T}}()
    for (qcharge, M) in mats
        F = qr(M)                              # `qr` here is LinearAlgebra.qr
        k = min(size(M)...)
        Qs[qcharge] = Matrix(F.Q)[:, 1:k]
        Rs[qcharge] = Matrix(F.R)[1:k, :]
    end
    Q = _left_factor(Qs, rgroups, row_legs, A)
    Rf = _right_factor(Rs, cgroups, col_legs, A)
    return Q, Rf
end

# ---------------------------------------------------------------------------
# Folding singular values into a factor
# ---------------------------------------------------------------------------

# Scale a block along dimension `d` by the vector `s`.
function _scale_dim(blk::Array{T,R}, s::AbstractVector, d::Int) where {T,R}
    shp = ntuple(i -> i == d ? length(s) : 1, R)
    return blk .* reshape(convert(Vector{T}, s), shp)
end

"""Fold singular values `S` (charge → values) into the outgoing-bond factor `V`
(bond is leg 1), scaling its bond rows. Used for a right-moving DMRG split."""
function absorb_S_right(S, V::SymTensor{T,R,K}) where {T,R,K}
    blocks = Dict{NTuple{R,Int},Array{T,R}}(
        key => _scale_dim(blk, S[V.legs[1].sectors[key[1]]], 1) for (key, blk) in V.blocks)
    return SymTensor{T,R,K}(V.legs, V.flux, blocks)
end

"""Fold singular values `S` into the incoming-bond factor `U` (bond is the last
leg), scaling its bond columns. Used for a left-moving DMRG split."""
function absorb_S_left(U::SymTensor{T,R,K}, S) where {T,R,K}
    blocks = Dict{NTuple{R,Int},Array{T,R}}(
        key => _scale_dim(blk, S[U.legs[R].sectors[key[R]]], R) for (key, blk) in U.blocks)
    return SymTensor{T,R,K}(U.legs, U.flux, blocks)
end
