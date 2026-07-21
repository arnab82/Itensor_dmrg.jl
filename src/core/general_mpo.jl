# Finite-range MPO compiler with operator "string" channels.
#
# `nearest_neighbor_mpo` only handles range-1, translation-invariant terms. This
# builder compiles an arbitrary, position-dependent sum of on-site terms and
# two-site terms of ANY range, where the two operators are joined by a string
# operator on the sites strictly between them:
#
#   H = Σ (coeff, site, Ô)                              # on-site
#     + Σ (coeff, sL, L̂, sR, R̂, Ŝ) with sL < sR         # two-site + string
#         ↦ coeff · L̂_{sL} · Ŝ_{sL+1} ⋯ Ŝ_{sR-1} · R̂_{sR}
#
# Setting `Ŝ = I` recovers a plain long-range term; `Ŝ = F` (fermion parity)
# gives a Jordan-Wigner fermionic term, which is what lattice-fermion hopping
# past intervening sites needs (e.g. vertical bonds of a snaked 2D lattice).
#
# Construction is the standard finite-state automaton. Bond states at each gap:
#   1                    = "done"  (a term has completed / trailing identity),
#   2 .. 1+c_g           = one open channel per two-site term crossing the gap,
#   2+c_g                = "start" (leading identity, nothing placed yet).
# Each term gets its own channel while open (no operator-algebra compression),
# so the bond dimension at a gap is 2 + (#terms spanning it) — for an Nx×Ny
# lattice this is O(Nx), the expected width scaling. Correct, not minimal.

# input term is (coeff, siteL, opL, siteR, opR, string); normalize to
# (sL, sR, coeff, L, R, S) with sL < sR
_norm_twosite(::Type{T}, d, term) where {T} = begin
    c, sL, L, sR, R, S = term
    sL, sR = Int(sL), Int(sR)
    sL < sR || throw(ArgumentError("two-site term needs sL < sR (got $sL, $sR)"))
    (sL, sR, convert(T, c), _as_op(T, d, L), _as_op(T, d, R), _as_op(T, d, S))
end

"""
    general_mpo(N, d; onsite=(), twosite=(), T=ComplexF64) -> MPO

Compile a length-`N`, physical-dimension-`d` open-chain Hamiltonian from a list
of on-site and finite-range two-site string terms into an MPO.

- `onsite`  : iterable of `(coeff, site, op)`.
- `twosite` : iterable of `(coeff, siteL, opL, siteR, opR, string_op)` with
  `siteL < siteR`; the term is
  `coeff · opL_{siteL} · string_op_{siteL+1..siteR-1} · opR_{siteR}`.

Use `string_op = I(d)` for a bosonic long-range term and `string_op = F` (the
fermion parity) for a Jordan-Wigner fermionic term.
"""
function general_mpo(N::Integer, d::Integer; onsite=(), twosite=(),
                     T::Type{<:Number}=ComplexF64)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    d >= 1 || throw(ArgumentError("d must be positive"))
    id = Matrix{T}(I, d, d)

    # accumulate on-site operators per site
    O = [zeros(T, d, d) for _ in 1:N]
    for (c, site, op) in onsite
        1 <= site <= N || throw(ArgumentError("onsite site $site out of range"))
        O[site] .+= convert(T, c) .* _as_op(T, d, op)
    end

    terms = [_norm_twosite(T, d, t) for t in twosite]
    for (sL, sR, _, _, _, _) in terms
        1 <= sL && sR <= N || throw(ArgumentError("two-site sites out of 1:$N"))
    end

    # open[g] = ordered list of term indices spanning gap g (between site g,g+1),
    # i.e. sL <= g < sR. Term k occupies gaps sL_k .. sR_k-1.
    open = [Int[] for _ in 0:N]          # index 1..N+1 for gaps 0..N (0 and N unused as internal)
    for (k, (sL, sR, _...)) in enumerate(terms)
        for g in sL:(sR-1)
            push!(open[g+1], k)          # gap g stored at open[g+1]
        end
    end
    gapopen(g) = open[g+1]

    # state counts and index maps per gap
    nstate(g) = (g == 0 || g == N) ? 1 : 2 + length(gapopen(g))
    done_idx(g) = 1                                   # valid at internal g and at gap N
    start_idx(g) = g == 0 ? 1 : 2 + length(gapopen(g))
    function chan_idx(g, k)
        p = findfirst(==(k), gapopen(g))
        p === nothing && error("term $k not open at gap $g")
        return 1 + p
    end

    tensors = Vector{Array{T,4}}(undef, N)
    for n in 1:N
        gl, gr = n - 1, n
        Ln, Rn = nstate(gl), nstate(gr)
        W = zeros(T, Ln, d, d, Rn)

        # identity propagation
        if gl != 0 && gr != 0            # done -> done  (done exists at internal gaps and gap N)
            W[done_idx(gl), :, :, done_idx(gr)] .+= id
        elseif gr == N && gl != 0
            W[done_idx(gl), :, :, done_idx(gr)] .+= id
        end
        if gl != N && gr != N            # start -> start (start exists at gap 0..N-1)
            W[start_idx(gl), :, :, start_idx(gr)] .+= id
        end
        # start -> done : on-site term at site n
        W[start_idx(gl), :, :, (gr == N ? done_idx(gr) : done_idx(gr))] .+= O[n]

        for (k, (sL, sR, c, L, R, S)) in enumerate(terms)
            if sL == n                    # open channel: start -> channel_k, emit c*L
                W[start_idx(gl), :, :, chan_idx(gr, k)] .+= c .* L
            end
            if sL < n < sR                # carry string: channel_k -> channel_k
                W[chan_idx(gl, k), :, :, chan_idx(gr, k)] .+= S
            end
            if sR == n                    # close channel: channel_k -> done, emit R
                W[chan_idx(gl, k), :, :, done_idx(gr)] .+= R
            end
        end
        tensors[n] = W
    end
    return MPO(tensors)
end
