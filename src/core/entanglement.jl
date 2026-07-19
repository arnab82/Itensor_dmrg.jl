# Bipartite entanglement of a dense MPS across a bond.
#
# The Schmidt values across the cut {1..b} | {b+1..N} are the singular values of
# the orthogonality-center tensor once the state is put in mixed-canonical form
# with its center at site b: sites 1..b-1 left-canonical, sites b+1..N
# right-canonical (see theory §3.4, §6). We work on a copy so the caller's MPS
# is untouched.

"""
    schmidt_values(psi, bond) -> Vector{Float64}

Schmidt (singular) values of the normalized state across the cut between sites
`bond` and `bond+1`, sorted in decreasing order. They satisfy
`sum(abs2, schmidt_values(psi, bond)) ≈ 1`, and their squares are the eigenvalues
of the reduced density matrix of either half.
"""
function schmidt_values(psi::MPS, bond::Integer)
    1 <= bond <= psi.N - 1 ||
        throw(ArgumentError("bond must be in 1:$(psi.N - 1)"))
    phi = copy(psi)
    normalize!(phi)
    right_canonicalize!(phi)              # sites 2..N right-canonical, center at site 1

    # Move the orthogonality center to site `bond` by left-canonicalizing 1..bond-1.
    for i in 1:bond-1
        A = phi.tensors[i]
        l, d, r = size(A)
        F = qr(reshape(A, l * d, r))
        q = min(l * d, r)
        Q = Matrix(F.Q)[:, 1:q]
        R = Matrix(F.R)[1:q, :]
        phi.tensors[i] = reshape(Q, l, d, q)
        B = phi.tensors[i + 1]
        phi.tensors[i + 1] = reshape(R * reshape(B, r, :), q, d, size(B, 3))
    end

    center = phi.tensors[bond]            # 1..bond-1 left-canonical, bond+1..N right-canonical
    l, d, r = size(center)
    return svdvals(reshape(center, l * d, r))
end

"""
    entanglement_entropy(psi, bond; base=ℯ) -> Float64

Von Neumann entanglement entropy `S = -Σ σₖ² log σₖ²` across the cut between
sites `bond` and `bond+1`, where `σₖ` are the Schmidt values. Pass `base=2` for
bits.
"""
function entanglement_entropy(psi::MPS, bond::Integer; base::Real=ℯ)
    s = schmidt_values(psi, bond)
    entropy = 0.0
    for σ in s
        p = σ^2
        p > 0 && (entropy -= p * log(p))
    end
    return entropy / log(base)
end

"""
    entanglement_entropy(psi; base=ℯ) -> Vector{Float64}

The entanglement-entropy profile `[S(bond) for bond in 1:N-1]` across every bond.
"""
entanglement_entropy(psi::MPS; base::Real=ℯ) =
    Float64[entanglement_entropy(psi, b; base=base) for b in 1:psi.N-1]
