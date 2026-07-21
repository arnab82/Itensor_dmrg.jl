# Block-sparse two-site DMRG for the `symmetry=true` path. It mirrors the dense
# engine in core/dmrg.jl but every contraction, eigensolve, and truncation runs
# block-by-block on `SymTensor`s, so the local eigenproblem is restricted to the
# target charge sector — the source of the memory/time win.
#
# Leg/arrow conventions (see sym_mps_mpo.jl): ket MPS A[l,s,r] = (+1,+1,-1);
# MPO W[l,s',s,r] = (+1,+1,-1,-1); left environment L[bra,mpo,ket] = (+1,-1,-1);
# right environment R[bra,mpo,ket] = (-1,+1,+1). Every tensor has flux 0 except
# the boundary environments, which pin the total charge on the right.

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

# Trivial rank-3 environment with a single unit block on the given leg charges.
function _boundary_env(::Type{T}, arrows, qs::NTuple{3,QN{K}}) where {T,K}
    legs = ntuple(i -> SymIndex(arrows[i], [qs[i]], [1]), 3)
    blocks = Dict((1, 1, 1) => reshape(T[1], 1, 1, 1))
    return SymTensor{T,3,K}(legs, zero(QN{K}), blocks)
end

left_boundary(::Type{T}, ::Val{K}) where {T,K} =
    _boundary_env(T, (+1, -1, -1), (zero(QN{K}), zero(QN{K}), zero(QN{K})))
right_boundary(::Type{T}, sector::QN{K}) where {T,K} =
    _boundary_env(T, (-1, +1, +1), (sector, zero(QN{K}), sector))

# Absorb one site into the left environment: L[bra,mpo,ket] -> L'[bra,mpo,ket].
function absorb_left_sym(L::SymTensor, A::SymTensor, W::SymTensor)
    bra = symconj(A)
    c1 = contract(L, bra, (1,), (1,))          # L.bra · bra.l
    c2 = contract(c1, W, (1, 3), (1, 2))       # .mpo·W.l , bra.s'·W.s'
    c3 = contract(c2, A, (1, 3), (1, 2))       # .ket·A.l , W.s·A.s
    return c3
end

# Absorb one site into the right environment: R[bra,mpo,ket] -> R'[bra,mpo,ket].
function absorb_right_sym(R::SymTensor, A::SymTensor, W::SymTensor)
    bra = symconj(A)
    c1 = contract(R, bra, (1,), (3,))          # R.bra · bra.r
    c2 = contract(c1, W, (1, 4), (4, 2))       # .mpo·W.r , bra.s'·W.s'
    c3 = contract(c2, A, (1, 4), (3, 2))       # .ket·A.r , W.s·A.s
    return c3
end

# ---------------------------------------------------------------------------
# Two-site tensor and the effective-Hamiltonian action
# ---------------------------------------------------------------------------

"""Merge sites `i, i+1` into a rank-4 two-site tensor `theta[l, s1, s2, r]`."""
two_site_sym(A1::SymTensor, A2::SymTensor) = contract(A1, A2, (3,), (1,))

"""Matrix-free action of the two-site effective Hamiltonian on `theta`."""
function sym_effective(theta::SymTensor, L::SymTensor, W1::SymTensor,
                       W2::SymTensor, R::SymTensor)
    t1 = contract(L, theta, (3,), (1,))        # L.ket · theta.l
    t2 = contract(t1, W1, (2, 3), (1, 3))      # L.mpo·W1.l , theta.s1·W1.s
    t3 = contract(t2, W2, (2, 5), (3, 1))      # theta.s2·W2.s , W1.r·W2.l
    Y = contract(t3, R, (2, 5), (3, 2))        # theta.r·R.ket , W2.r·R.mpo
    return Y                                    # legs identical to theta
end

# Flat packing of a two-site tensor's charge-allowed block amplitudes, so the
# eigensolve runs on a plain Vector of exactly the target-sector degrees of
# freedom.
struct ThetaLayout{R,K}
    legs::NTuple{R,SymIndex{K}}
    flux::QN{K}
    keys::Vector{NTuple{R,Int}}
    dims::Vector{NTuple{R,Int}}
    offs::Vector{Int}
    len::Int
end

function theta_layout(legs::NTuple{R,SymIndex{K}}, flux::QN{K}) where {R,K}
    keys = sort(_allowed_keys(legs, flux))
    dims = [_blockdims(legs, k) for k in keys]
    offs = Int[]
    o = 0
    for d in dims
        push!(offs, o)
        o += prod(d)
    end
    return ThetaLayout{R,K}(legs, flux, keys, dims, offs, o)
end

function pack(A::SymTensor{T,R,K}, lay::ThetaLayout{R,K}) where {T,R,K}
    v = zeros(T, lay.len)
    for (i, k) in enumerate(lay.keys)
        blk = get(A.blocks, k, nothing)
        blk === nothing && continue
        v[lay.offs[i] + 1:lay.offs[i] + prod(lay.dims[i])] = vec(blk)
    end
    return v
end

function unpack(v::AbstractVector{T}, lay::ThetaLayout{R,K}) where {T,R,K}
    blocks = Dict{NTuple{R,Int},Array{T,R}}()
    for (i, k) in enumerate(lay.keys)
        rng = lay.offs[i] + 1:lay.offs[i] + prod(lay.dims[i])
        blocks[k] = reshape(Array{T}(v[rng]), lay.dims[i])
    end
    return SymTensor{T,R,K}(lay.legs, lay.flux, blocks)
end

# ---------------------------------------------------------------------------
# Local eigensolve and two-site split
# ---------------------------------------------------------------------------

function lowest_local_sym(theta::SymTensor{T}, L, W1, W2, R;
                          eig_tol::Real=1e-10, krylovdim::Integer=30,
                          maxiter::Integer=200) where {T}
    lay = theta_layout(theta.legs, theta.flux)
    lay.len == 0 && throw(ArgumentError("empty local problem (sector has no states here)"))
    v0 = pack(theta, lay)
    action = v -> pack(sym_effective(unpack(v, lay), L, W1, W2, R), lay)
    kd = min(lay.len, max(2, krylovdim))
    vals, vecs, info = KrylovKit.eigsolve(action, v0, 1, :SR;
                                          ishermitian=true, tol=eig_tol,
                                          krylovdim=kd, maxiter=maxiter)
    return real(vals[1]), unpack(vecs[1], lay), info
end

# Split a two-site tensor and move the orthogonality centre. Returns the new
# (A_i, A_{i+1}) and the discarded weight.
function split_two_site_sym(theta::SymTensor, direction::Symbol,
                            maxdim::Integer, cutoff::Real)
    U, S, V, discarded = svd_truncated(theta, (1, 2), (3, 4); maxdim=maxdim, cutoff=cutoff)
    if direction === :right
        return U, absorb_S_right(S, V), discarded          # centre moves right
    elseif direction === :left
        return absorb_S_left(U, S), V, discarded           # centre moves left
    else
        throw(ArgumentError("direction must be :right or :left"))
    end
end

# ---------------------------------------------------------------------------
# Canonicalization and normalization (via the untruncated block SVD)
# ---------------------------------------------------------------------------

const _NOTRUNC = typemax(Int)

"""Right-canonicalize in place: sites `2..N` become right-isometries."""
function right_canonicalize_sym!(psi::SymMPS)
    for i in psi.N:-1:2
        U, S, V, _ = svd_truncated(psi.tensors[i], (1,), (2, 3); maxdim=_NOTRUNC, cutoff=0.0)
        psi.tensors[i] = V                                         # right-isometric
        psi.tensors[i - 1] = contract(psi.tensors[i - 1], absorb_S_left(U, S), (3,), (1,))
    end
    return psi
end

function normalize_sym!(psi::SymMPS)
    nrm = norm(psi.tensors[1])
    nrm > eps(Float64) || throw(ArgumentError("cannot normalize a zero MPS"))
    psi.tensors[1] = (1 / nrm) * psi.tensors[1]
    return psi
end

# ---------------------------------------------------------------------------
# Sweeps and the driver
# ---------------------------------------------------------------------------

function _build_right_envs(H::SymMPO, psi::SymMPS, ::Type{T}) where {T}
    N = psi.N
    R = Vector{SymTensor{T}}(undef, N + 1)
    R[N + 1] = right_boundary(T, psi.sector)
    for i in N:-1:1
        R[i] = absorb_right_sym(R[i + 1], psi.tensors[i], H.tensors[i])
    end
    return R
end

function sym_sweep!(H::SymMPO, psi::SymMPS, Lenv, Renv, direction::Symbol;
                    maxdim::Integer, cutoff::Real, eig_tol::Real)
    N = psi.N
    sites = direction === :right ? (1:N - 1) : (N - 1:-1:1)
    energy = Inf
    discarded = 0.0
    for i in sites
        theta = two_site_sym(psi.tensors[i], psi.tensors[i + 1])
        energy, theta_gs, _ = lowest_local_sym(theta, Lenv[i], H.tensors[i],
                                               H.tensors[i + 1], Renv[i + 2]; eig_tol=eig_tol)
        A1, A2, disc = split_two_site_sym(theta_gs, direction, maxdim, cutoff)
        discarded += disc
        psi.tensors[i] = A1
        psi.tensors[i + 1] = A2
        if direction === :right
            Lenv[i + 1] = absorb_left_sym(Lenv[i], psi.tensors[i], H.tensors[i])
        else
            Renv[i + 1] = absorb_right_sym(Renv[i + 2], psi.tensors[i + 1], H.tensors[i + 1])
        end
    end
    return energy, discarded
end

"""
    sym_dmrg!(H::SymMPO, psi::SymMPS; nsweeps, maxdim, cutoff, tol, eig_tol, output)

Block-sparse two-site DMRG, mutating `psi`. `psi` must be pinned to the desired
charge sector (see `random_MPS(H::SymMPO; sector=...)`).
"""
function sym_dmrg!(H::SymMPO, psi::SymMPS; nsweeps::Integer=20, maxdim=100,
                   cutoff=1e-10, tol=1e-8, eig_tol=1e-10, output::Bool=true)
    (H.N == psi.N) || throw(DimensionMismatch("MPO and MPS lengths differ"))
    T = promote_type(eltype(H), eltype(psi))
    right_canonicalize_sym!(psi)
    normalize_sym!(psi)

    Lenv = Vector{SymTensor{T}}(undef, psi.N + 1)
    Lenv[1] = left_boundary(T, Val(length(psi.sector)))
    Renv = _build_right_envs(H, psi, T)

    previous = Inf
    energy = Inf
    for sweep in 1:nsweeps
        md = Int(schedule_value(maxdim, sweep, :maxdim))
        ct = Float64(schedule_value(cutoff, sweep, :cutoff))
        tl = Float64(schedule_value(tol, sweep, :tol))
        et = Float64(schedule_value(eig_tol, sweep, :eig_tol))

        Renv = _build_right_envs(H, psi, T)                # refresh before L→R
        energy, err_r = sym_sweep!(H, psi, Lenv, Renv, :right; maxdim=md, cutoff=ct, eig_tol=et)
        energy, err_l = sym_sweep!(H, psi, Lenv, Renv, :left; maxdim=md, cutoff=ct, eig_tol=et)

        delta = abs(energy - previous)
        output && @printf("NaiveDMRG[sym] sweep %d: energy = %.12f  delta = %.3e  discarded = %.3e\n",
                          sweep, energy, delta, err_r + err_l)
        delta <= tl && return energy, psi
        previous = energy
    end
    return energy, psi
end

"""Run block-sparse DMRG (dispatches on the symmetric types)."""
dmrg(H::SymMPO, psi::SymMPS; kwargs...) = sym_dmrg!(H, deepcopy_sym(psi); kwargs...)

deepcopy_sym(psi::SymMPS{T,K}) where {T,K} =
    SymMPS{T,K}([SymTensor{T,3,K}(t.legs, t.flux, Dict(k => copy(v) for (k, v) in t.blocks))
                 for t in psi.tensors], psi.sites, psi.sector, psi.N, psi.d)
