mutable struct EnvironmentCache
    left::Vector{Array{ComplexF64,3}}
    right::Vector{Array{ComplexF64,3}}
end

"""Diagnostics from one local two-site Krylov eigensolve."""
struct LocalSolveRecord{T<:Real}
    site::Int
    direction::Symbol
    energy::T
    converged::Bool
    residual_norm::T
    iterations::Int
    operations::Int
end


"""Diagnostics collected after one complete left-and-right sweep."""
struct SweepRecord{T<:Real}
    sweep::Int
    energy::T
    energy_change::T
    discarded_weight::T
    max_bond_dimension::Int
    maxdim_limit::Int
    cutoff::T
    tolerance::T
    eigensolver_tolerance::T
    local_solves::Vector{LocalSolveRecord{T}}
end


"""Result of a DMRG run, including the optimized state and convergence history."""
struct DMRGResult{T<:Real,S<:MPS}
    energy::T
    state::S
    converged::Bool
    stopping_reason::Symbol
    history::Vector{SweepRecord{T}}
end

# Preserve convenient `energy, psi = dmrg(...)` destructuring.
Base.length(::DMRGResult) = 2
Base.iterate(result::DMRGResult) = (result.energy, 2)
Base.iterate(result::DMRGResult, state::Int) = state == 2 ? (result.state, 3) : nothing
Base.getindex(result::DMRGResult, i::Int) = i == 1 ? result.energy : i == 2 ? result.state : throw(BoundsError(result, i))

function EnvironmentCache(N::Integer)
    boundary = ones(ComplexF64, 1, 1, 1)
    return EnvironmentCache([copy(boundary) for _ in 0:N], [copy(boundary) for _ in 0:N])
end

function absorb_left(L, A, W)
    @tensor next_L[rb, b, rk] := conj(A[lb, so, rb]) * L[lb, a, lk] *
                                  W[a, so, si, b] * A[lk, si, rk]
    return next_L
end

function absorb_left!(out, L, A, W)
    expected = (size(A, 3), size(W, 4), size(A, 3))
    size(out) == expected || throw(DimensionMismatch("left environment buffer has the wrong shape"))
    fill!(out, zero(eltype(out)))
    @tensor out[rb, b, rk] = conj(A[lb, so, rb]) * L[lb, a, lk] *
                             W[a, so, si, b] * A[lk, si, rk]
    return out
end

function absorb_right(R, A, W)
    @tensor next_R[lb, a, lk] := conj(A[lb, so, rb]) * W[a, so, si, b] *
                                  A[lk, si, rk] * R[rb, b, rk]
    return next_R
end

function absorb_right!(out, R, A, W)
    expected = (size(A, 1), size(W, 1), size(A, 1))
    size(out) == expected || throw(DimensionMismatch("right environment buffer has the wrong shape"))
    fill!(out, zero(eltype(out)))
    @tensor out[lb, a, lk] = conj(A[lb, so, rb]) * W[a, so, si, b] *
                             A[lk, si, rk] * R[rb, b, rk]
    return out
end

function environment_buffer!(buffers, index, shape)
    if size(buffers[index]) != shape
        buffers[index] = zeros(ComplexF64, shape)
    end
    return buffers[index]
end

function environments(H::MPO, psi::MPS)
    (H.N, H.d) == (psi.N, psi.d) || throw(DimensionMismatch("MPO and MPS are incompatible"))
    cache = EnvironmentCache(psi.N)
    for i in 1:psi.N
        cache.left[i + 1] = absorb_left(cache.left[i], psi.tensors[i], H.tensors[i])
    end
    for i in psi.N:-1:1
        cache.right[i] = absorb_right(cache.right[i + 1], psi.tensors[i], H.tensors[i])
    end
    return cache
end

"""Build only the right environments needed to begin a left-to-right sweep."""
function right_sweep_cache!(cache::EnvironmentCache, H::MPO, psi::MPS)
    (H.N, H.d) == (psi.N, psi.d) || throw(DimensionMismatch("MPO and MPS are incompatible"))
    cache.left[1] .= one(ComplexF64)
    cache.right[psi.N + 1] .= one(ComplexF64)
    # Pair i,i+1 consumes right[i+2]. Sites 1 and 2 are therefore never part
    # of a precomputed right environment for a right-moving two-site sweep.
    for site in psi.N:-1:3
        A, W = psi.tensors[site], H.tensors[site]
        out = environment_buffer!(cache.right, site, (size(A, 1), size(W, 1), size(A, 1)))
        absorb_right!(out, cache.right[site + 1], A, W)
    end
    return cache
end


right_sweep_cache(H::MPO, psi::MPS) = right_sweep_cache!(EnvironmentCache(psi.N), H, psi)

function two_site_tensor(psi::MPS, i::Integer)
    A, B = psi.tensors[i], psi.tensors[i + 1]
    @tensor theta[l, s1, s2, r] := A[l, s1, m] * B[m, s2, r]
    return theta
end

"""Matrix-free action of the two-site effective Hamiltonian."""
function effective_action(x, L, W1, W2, R, shape)
    X = reshape(x, shape)
    @tensor Y[lb, so1, so2, rb] := L[lb, a, lk] * W1[a, so1, si1, m] *
                                      W2[m, so2, si2, b] * R[rb, b, rk] *
                                      X[lk, si1, si2, rk]
    return vec(Y)
end

function effective_action!(y, x, L, W1, W2, R, shape)
    length(y) == prod(shape) || throw(DimensionMismatch("effective-action buffer has the wrong length"))
    X = reshape(x, shape)
    Y = reshape(y, shape)
    fill!(Y, zero(eltype(Y)))
    @tensor Y[lb, so1, so2, rb] = L[lb, a, lk] * W1[a, so1, si1, m] *
                                  W2[m, so2, si2, b] * R[rb, b, rk] *
                                  X[lk, si1, si2, rk]
    return y
end

function lowest_local_state(H::MPO, psi::MPS, cache::EnvironmentCache, i::Integer;
                            direction::Symbol, eig_tol::Real=1e-10,
                            krylovdim::Integer=30, maxiter::Integer=200)
    theta = two_site_tensor(psi, i)
    shape = size(theta)
    action = x -> effective_action(x, cache.left[i], H.tensors[i], H.tensors[i + 1],
                                   cache.right[i + 2], shape)
    dim = length(theta)
    kd = min(dim, max(2, krylovdim))
    values, vectors, info = KrylovKit.eigsolve(action, vec(theta), 1, :SR;
                                               ishermitian=true, tol=eig_tol,
                                               krylovdim=kd, maxiter=maxiter)
    converged = info.converged >= 1
    residual_norm = real(info.normres[1])
    converged || @warn "local eigensolver did not converge" site=i direction residual=residual_norm
    energy = real(values[1])
    record = LocalSolveRecord(i, direction, energy, converged, residual_norm,
                              info.numiter, info.numops)
    return energy, reshape(vectors[1], shape), record
end

function kept_dimension(s::AbstractVector, maxdim::Integer, cutoff::Real)
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    keep = min(maxdim, length(s))
    if cutoff > 0
        total = sum(abs2, s)
        discarded = 0.0
        while keep > 1 && discarded + abs2(s[keep]) <= cutoff * total
            discarded += abs2(s[keep])
            keep -= 1
        end
    end
    return keep
end

function split_two_site!(psi::MPS, i::Integer, theta, direction::Symbol,
                         maxdim::Integer, cutoff::Real)
    l, d1, d2, r = size(theta)
    F = svd(reshape(theta, l * d1, d2 * r))
    keep = kept_dimension(F.S, maxdim, cutoff)
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    discarded = keep == length(F.S) ? 0.0 : sum(abs2, @view F.S[keep + 1:end])

    if direction === :right
        psi.tensors[i] = reshape(U, l, d1, keep)
        psi.tensors[i + 1] = reshape(Diagonal(S) * Vt, keep, d2, r)
    elseif direction === :left
        psi.tensors[i] = reshape(U * Diagonal(S), l, d1, keep)
        psi.tensors[i + 1] = reshape(Vt, keep, d2, r)
    else
        throw(ArgumentError("direction must be :right or :left"))
    end
    return discarded
end

function sweep!(H::MPO, psi::MPS, cache::EnvironmentCache, direction::Symbol;
                maxdim::Integer, cutoff::Real, eig_tol::Real)
    sites = direction === :right ? (1:psi.N-1) : (psi.N-1:-1:1)
    local_energy = Inf
    discarded = 0.0
    records = LocalSolveRecord{Float64}[]

    for i in sites
        local_energy, theta, record = lowest_local_state(
            H, psi, cache, i; direction=direction, eig_tol=eig_tol
        )
        push!(records, record)
        discarded += split_two_site!(psi, i, theta, direction, maxdim, cutoff)
        if direction === :right
            A, W = psi.tensors[i], H.tensors[i]
            out = environment_buffer!(cache.left, i + 1, (size(A, 3), size(W, 4), size(A, 3)))
            absorb_left!(out, cache.left[i], A, W)
        else
            A, W = psi.tensors[i + 1], H.tensors[i + 1]
            out = environment_buffer!(cache.right, i + 1, (size(A, 1), size(W, 1), size(A, 1)))
            absorb_right!(out, cache.right[i + 2], A, W)
        end
    end
    return local_energy, discarded, records
end

function overlap_with_operator(bra::MPS, H::MPO, ket::MPS)
    (bra.N, bra.d) == (ket.N, ket.d) == (H.N, H.d) ||
        throw(DimensionMismatch("MPO and MPS dimensions differ"))
    env = ones(ComplexF64, 1, 1, 1)
    for i in 1:H.N
        A, W, B = bra.tensors[i], H.tensors[i], ket.tensors[i]
        @tensor next_env[ra, b, rk] := conj(A[la, so, ra]) * env[la, a, lk] *
                                       W[a, so, si, b] * B[lk, si, rk]
        env = next_env
    end
    return env[]
end

"""Normalized variational energy `⟨psi|H|psi⟩ / ⟨psi|psi⟩`."""
function compute_energy(H::MPO, psi::MPS)
    denominator = overlap(psi, psi)
    abs(denominator) > eps(Float64) || throw(ArgumentError("zero-norm MPS"))
    value = overlap_with_operator(psi, H, psi) / denominator
    abs(imag(value)) <= 1e-9 * max(1, abs(real(value))) ||
        @warn "energy has a non-negligible imaginary part" value
    return real(value)
end


schedule_value(value::Number, ::Int, ::Symbol) = value
function schedule_value(values, sweep::Int, name::Symbol)
    values isa Tuple || values isa AbstractVector ||
        throw(ArgumentError("$name must be a number, tuple, or vector"))
    isempty(values) && throw(ArgumentError("$name schedule cannot be empty"))
    return values[min(sweep, length(values))]
end

"""
    dmrg!(H, psi; nsweeps=20, maxdim=100, cutoff=1e-10, tol=1e-8, eig_tol=1e-10, output=true)

Run the from-scratch two-site, finite-system DMRG, mutating `psi` in place.
ITensor is not used by this solver.
"""
function dmrg!(H::MPO, psi::MPS; nsweeps::Integer=20, maxdim=100,
               cutoff=1e-10, tol=1e-8, eig_tol=1e-10,
               output::Bool=true)
    (H.N, H.d) == (psi.N, psi.d) || throw(DimensionMismatch("MPO and MPS are incompatible"))
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    right_canonicalize!(psi)
    normalize!(psi)
    previous = compute_energy(H, psi)

    history = SweepRecord{Float64}[]
    cache = EnvironmentCache(psi.N)
    for sweep in 1:nsweeps
        sweep_maxdim = Int(schedule_value(maxdim, sweep, :maxdim))
        sweep_cutoff = Float64(schedule_value(cutoff, sweep, :cutoff))
        sweep_tol = Float64(schedule_value(tol, sweep, :tol))
        sweep_eig_tol = Float64(schedule_value(eig_tol, sweep, :eig_tol))
        sweep_maxdim >= 1 || throw(ArgumentError("maxdim values must be positive"))
        sweep_cutoff >= 0 || throw(ArgumentError("cutoff values must be nonnegative"))
        sweep_tol >= 0 || throw(ArgumentError("tol values must be nonnegative"))
        sweep_eig_tol > 0 || throw(ArgumentError("eig_tol values must be positive"))

        right_sweep_cache!(cache, H, psi)
        _, error_right, right_records = sweep!(
            H, psi, cache, :right; maxdim=sweep_maxdim, cutoff=sweep_cutoff,
            eig_tol=sweep_eig_tol
        )
        _, error_left, left_records = sweep!(
            H, psi, cache, :left; maxdim=sweep_maxdim, cutoff=sweep_cutoff,
            eig_tol=sweep_eig_tol
        )
        energy = compute_energy(H, psi)
        delta = abs(energy - previous)
        record = SweepRecord(sweep, energy, delta, error_right + error_left,
                             maximum(bond_dimensions(psi)), sweep_maxdim,
                             sweep_cutoff, sweep_tol, sweep_eig_tol,
                             vcat(right_records, left_records))
        push!(history, record)
        output && @printf("NaiveDMRG sweep %d: energy = %.12f  delta = %.3e  discarded = %.3e\n",
                          sweep, energy, delta, error_right + error_left)
        if delta <= sweep_tol
            normalize!(psi)
            return DMRGResult(energy, psi, true, :energy_tolerance, history)
        end
        previous = energy
    end
    normalize!(psi)
    energy = compute_energy(H, psi)
    return DMRGResult(energy, psi, false, :maximum_sweeps, history)
end

"""Run DMRG on a copy of `psi`, leaving the supplied state unchanged."""
dmrg(H::MPO, psi::MPS; kwargs...) = dmrg!(H, copy(psi); kwargs...)

# Backward-compatible positional interfaces. The former `hubbard` flag never
# belonged in the generic sweep algorithm, so it is accepted but ignored.
dmrg!(H::MPO, psi::MPS, nsweeps::Integer, maxdim::Integer, tol::Real, hubbard::Bool=false) =
    dmrg!(H, psi; nsweeps=nsweeps, maxdim=maxdim, tol=tol)
dmrg(H::MPO, psi::MPS, nsweeps::Integer, maxdim::Integer, tol::Real, hubbard::Bool=false) =
    dmrg(H, psi; nsweeps=nsweeps, maxdim=maxdim, tol=tol)
