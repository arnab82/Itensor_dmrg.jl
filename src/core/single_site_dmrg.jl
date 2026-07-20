# Single-site DMRG with subspace expansion (a "DMRG3S"-style algorithm).
#
# Two-site DMRG grows bond dimensions naturally through the SVD of a merged
# two-site tensor. Single-site DMRG optimizes one site tensor at a time, which
# is cheaper but cannot, on its own, change a bond dimension — and so gets stuck
# in local minima. Subspace expansion fixes this: before moving the orthogonality
# center, the active bond is enlarged with an H-informed perturbation built from
# the environment and MPO, then SVD-truncated toward `maxdim`. The mixing factor
# `alpha` is decayed to zero so late sweeps converge as exact single-site DMRG.
#
# Reuses the environment machinery and helpers defined in dmrg.jl.

"""Matrix-free action of the single-site effective Hamiltonian on site tensor `x`."""
function effective_action_1site(x, L, W, R, shape)
    X = reshape(x, shape)
    @tensor Y[l, so, r] := L[l, a, lk] * W[a, so, si, b] * R[r, b, rk] * X[lk, si, rk]
    return vec(Y)
end

function lowest_local_state_1site(H::MPO, psi::MPS, cache::EnvironmentCache, i::Integer;
                                  direction::Symbol, eig_tol::Real=1e-10,
                                  krylovdim::Integer=30, maxiter::Integer=200)
    A = psi.tensors[i]
    shape = size(A)
    L, W, R = cache.left[i], H.tensors[i], cache.right[i + 1]
    action = x -> effective_action_1site(x, L, W, R, shape)
    kd = min(length(A), max(2, krylovdim))
    values, vectors, info = KrylovKit.eigsolve(action, vec(A), 1, :SR;
                                               ishermitian=true, tol=eig_tol,
                                               krylovdim=kd, maxiter=maxiter)
    converged = info.converged >= 1
    residual_norm = real(info.normres[1])
    converged || @warn "single-site local eigensolver did not converge" site=i direction residual=residual_norm
    energy = real(values[1])
    record = LocalSolveRecord(i, direction, energy, converged, residual_norm,
                              info.numiter, info.numops)
    return energy, reshape(vectors[1], shape), record
end

# Right move (center i → i+1): enlarge the right bond of `M` with the
# perturbation P[l,so,(b,r)] = Σ L[l,a,lk] W[a,so,si,b] M[lk,si,r], pad the next
# site with zeros so the state is unchanged, then SVD-truncate. `M` becomes
# left-canonical `U`; `S V†` is folded into the (enlarged) next site.
function expand_split_right!(psi::MPS, H::MPO, cache::EnvironmentCache, i::Integer,
                            M, alpha::Real, maxdim::Integer, cutoff::Real)
    l, d, r = size(M)
    if alpha > 0
        L, W = cache.left[i], H.tensors[i]
        w = size(W, 4)
        @tensor Pfull[l, so, b, r] := L[l, a, lk] * W[a, so, si, b] * M[lk, si, r]
        P = reshape(Pfull, l, d, w * r)
        Menl = cat(M, alpha .* P; dims=3)                 # (l, d, r + w*r)
        B = psi.tensors[i + 1]
        Benl = zeros(eltype(M), r + w * r, size(B, 2), size(B, 3))
        Benl[1:r, :, :] .= B
    else
        Menl, Benl = M, psi.tensors[i + 1]
    end
    rM = size(Menl, 3)
    F = svd(reshape(Menl, l * d, rM))
    keep = kept_dimension(F.S, maxdim, cutoff)
    discarded = keep == length(F.S) ? 0.0 : sum(abs2, @view F.S[keep + 1:end])
    psi.tensors[i] = reshape(F.U[:, 1:keep], l, d, keep)  # left-canonical
    fold = Diagonal(F.S[1:keep]) * F.Vt[1:keep, :]        # (keep, rM)
    newB = fold * reshape(Benl, rM, :)
    psi.tensors[i + 1] = reshape(newB, keep, size(Benl, 2), size(Benl, 3))
    return discarded
end

# Left move (center i → i-1): mirror image. Enlarge the left bond of `M` with
# P[(a,l),so,r] = Σ R[r,b,rk] W[a,so,si,b] M[l,si,rk]. `M` becomes right-canonical
# `V†`; `U S` is folded into the (enlarged) previous site.
function expand_split_left!(psi::MPS, H::MPO, cache::EnvironmentCache, i::Integer,
                           M, alpha::Real, maxdim::Integer, cutoff::Real)
    l, d, r = size(M)
    if alpha > 0
        R, W = cache.right[i + 1], H.tensors[i]
        wl = size(W, 1)
        @tensor Pfull[a, l, so, r] := R[r, b, rk] * W[a, so, si, b] * M[l, si, rk]
        P = reshape(Pfull, wl * l, d, r)
        Menl = cat(M, alpha .* P; dims=1)                 # (l + wl*l, d, r)
        A = psi.tensors[i - 1]
        Aenl = zeros(eltype(M), size(A, 1), size(A, 2), l + wl * l)
        Aenl[:, :, 1:l] .= A
    else
        Menl, Aenl = M, psi.tensors[i - 1]
    end
    lM = size(Menl, 1)
    F = svd(reshape(Menl, lM, d * r))
    keep = kept_dimension(F.S, maxdim, cutoff)
    discarded = keep == length(F.S) ? 0.0 : sum(abs2, @view F.S[keep + 1:end])
    psi.tensors[i] = reshape(F.Vt[1:keep, :], keep, d, r)  # right-canonical
    fold = F.U[:, 1:keep] * Diagonal(F.S[1:keep])          # (lM, keep)
    newA = reshape(Aenl, :, lM) * fold
    psi.tensors[i - 1] = reshape(newA, size(Aenl, 1), size(Aenl, 2), keep)
    return discarded
end

function sweep_1site!(H::MPO, psi::MPS, direction::Symbol;
                      maxdim::Integer, cutoff::Real, eig_tol::Real, alpha::Real)
    cache = environments(H, psi)
    local_energy = Inf
    discarded = 0.0
    records = LocalSolveRecord{Float64}[]
    if direction === :right
        for i in 1:psi.N-1
            local_energy, M, record = lowest_local_state_1site(
                H, psi, cache, i; direction=:right, eig_tol=eig_tol)
            push!(records, record)
            discarded += expand_split_right!(psi, H, cache, i, M, alpha, maxdim, cutoff)
            A, W = psi.tensors[i], H.tensors[i]
            out = environment_buffer!(cache.left, i + 1, (size(A, 3), size(W, 4), size(A, 3)))
            absorb_left!(out, cache.left[i], A, W)
        end
    else
        for i in psi.N:-1:2
            local_energy, M, record = lowest_local_state_1site(
                H, psi, cache, i; direction=:left, eig_tol=eig_tol)
            push!(records, record)
            discarded += expand_split_left!(psi, H, cache, i, M, alpha, maxdim, cutoff)
            A, W = psi.tensors[i], H.tensors[i]
            out = environment_buffer!(cache.right, i, (size(A, 1), size(W, 1), size(A, 1)))
            absorb_right!(out, cache.right[i + 1], A, W)
        end
    end
    return local_energy, discarded, records
end

"""
    single_site_dmrg!(H, psi; nsweeps=20, maxdim=100, cutoff=1e-10, tol=1e-8,
                      eig_tol=1e-10, alpha=(1e-2, 1e-3, 1e-4, 0.0), output=true)

Single-site, finite-system DMRG with subspace expansion, mutating `psi` in place.
Each of `maxdim`, `cutoff`, `tol`, `eig_tol`, and the expansion factor `alpha`
accepts a scalar or a per-sweep schedule. Decay `alpha` to `0` so late sweeps
converge as exact single-site DMRG; convergence is only declared once
`alpha == 0`. Returns a [`DMRGResult`](@ref). ITensor is not used.
"""
function single_site_dmrg!(H::MPO, psi::MPS; nsweeps::Integer=20, maxdim=100,
                           cutoff=1e-10, tol=1e-8, eig_tol=1e-10,
                           alpha=(1e-2, 1e-3, 1e-4, 0.0), output::Bool=true)
    (H.N, H.d) == (psi.N, psi.d) || throw(DimensionMismatch("MPO and MPS are incompatible"))
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    T = promote_type(eltype(H), eltype(psi))
    T == eltype(psi) || throw(ArgumentError(
        "psi has scalar type $(eltype(psi)) but the Hamiltonian requires $T; " *
        "use the copying `single_site_dmrg`, or convert psi first"))
    right_canonicalize!(psi)
    normalize!(psi)
    previous = compute_energy(H, psi)

    history = SweepRecord{Float64}[]
    for sweep in 1:nsweeps
        sweep_maxdim = Int(schedule_value(maxdim, sweep, :maxdim))
        sweep_cutoff = Float64(schedule_value(cutoff, sweep, :cutoff))
        sweep_tol = Float64(schedule_value(tol, sweep, :tol))
        sweep_eig_tol = Float64(schedule_value(eig_tol, sweep, :eig_tol))
        sweep_alpha = Float64(schedule_value(alpha, sweep, :alpha))
        sweep_maxdim >= 1 || throw(ArgumentError("maxdim values must be positive"))
        sweep_cutoff >= 0 || throw(ArgumentError("cutoff values must be nonnegative"))
        sweep_tol >= 0 || throw(ArgumentError("tol values must be nonnegative"))
        sweep_eig_tol > 0 || throw(ArgumentError("eig_tol values must be positive"))
        sweep_alpha >= 0 || throw(ArgumentError("alpha values must be nonnegative"))

        _, error_right, right_records = sweep_1site!(
            H, psi, :right; maxdim=sweep_maxdim, cutoff=sweep_cutoff,
            eig_tol=sweep_eig_tol, alpha=sweep_alpha)
        _, error_left, left_records = sweep_1site!(
            H, psi, :left; maxdim=sweep_maxdim, cutoff=sweep_cutoff,
            eig_tol=sweep_eig_tol, alpha=sweep_alpha)
        normalize!(psi)
        energy = compute_energy(H, psi)
        delta = abs(energy - previous)
        record = SweepRecord(sweep, energy, delta, error_right + error_left,
                             maximum(bond_dimensions(psi)), sweep_maxdim,
                             sweep_cutoff, sweep_tol, sweep_eig_tol,
                             vcat(right_records, left_records))
        push!(history, record)
        output && @printf("NaiveDMRG single-site sweep %d: energy = %.12f  delta = %.3e  discarded = %.3e  alpha = %.1e  maxχ = %d\n",
                          sweep, energy, delta, error_right + error_left,
                          sweep_alpha, maximum(bond_dimensions(psi)))
        if delta <= sweep_tol && sweep_alpha == 0.0
            return DMRGResult(energy, psi, true, :energy_tolerance, history)
        end
        previous = energy
    end
    normalize!(psi)
    energy = compute_energy(H, psi)
    return DMRGResult(energy, psi, false, :maximum_sweeps, history)
end

"""
Run single-site DMRG on a copy of `psi`, leaving the supplied state unchanged.
The copy is promoted to `promote_type(eltype(H), eltype(psi))`.
"""
single_site_dmrg(H::MPO, psi::MPS; kwargs...) =
    single_site_dmrg!(H, _promote_mps(psi, promote_type(eltype(H), eltype(psi))); kwargs...)
