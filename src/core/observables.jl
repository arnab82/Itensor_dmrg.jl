# Local observables and correlation functions for a dense MPS.
#
# Every quantity is the normalized expectation value ⟨ψ|Ô|ψ⟩ / ⟨ψ|ψ⟩, evaluated
# by sweeping a two-index MPS-MPS environment [bra bond, ket bond] along the
# chain and inserting the local operator(s) at the requested site(s). A local
# operator is any d×d matrix ⟨s'|Ô|s⟩ = O[s', s].
#
# Results are returned as `ComplexF64` for type stability and to support
# non-Hermitian operators (e.g. S⁺, needed by ⟨S⁺ᵢ S⁻ⱼ⟩). For a Hermitian
# operator the imaginary part is zero up to rounding — take `real(...)`.

"""
    spin_half_operators()

Return the spin-1/2 operators as 2×2 `ComplexF64` matrices in the `{|↑⟩, |↓⟩}`
basis, as a `NamedTuple` with fields `Sx, Sy, Sz, Sp, Sm, Id`.
"""
function spin_half_operators()
    return (
        Sx = ComplexF64[0 0.5; 0.5 0],
        Sy = ComplexF64[0 -0.5im; 0.5im 0],
        Sz = ComplexF64[0.5 0; 0 -0.5],
        Sp = ComplexF64[0 1; 0 0],
        Sm = ComplexF64[0 0; 1 0],
        Id = ComplexF64[1 0; 0 1],
    )
end

# Grow the MPS-MPS environment across one site, optionally inserting operator O.
@inline function _absorb_site(E, A, O)
    if O === nothing
        @tensor Enext[ra, rb] := conj(A[la, s, ra]) * E[la, lb] * A[lb, s, rb]
    else
        @tensor Enext[ra, rb] := conj(A[la, sp, ra]) * E[la, lb] * O[sp, s] * A[lb, s, rb]
    end
    return Enext
end

_check_op(psi::MPS, op::AbstractMatrix) =
    size(op) == (psi.d, psi.d) ||
        throw(DimensionMismatch("operator must be $(psi.d)×$(psi.d) to match the physical dimension"))

"""
    expect(psi, op, site) -> ComplexF64

Normalized single-site expectation value `⟨ψ|Ô_site|ψ⟩ / ⟨ψ|ψ⟩` of the local
`d×d` operator `op` acting on `site`.
"""
function expect(psi::MPS, op::AbstractMatrix, site::Integer)
    1 <= site <= psi.N || throw(BoundsError(psi, site))
    _check_op(psi, op)
    O = Array{ComplexF64}(op)
    E = ones(ComplexF64, 1, 1)
    for k in 1:psi.N
        E = _absorb_site(E, psi.tensors[k], k == site ? O : nothing)
    end
    return E[] / real(overlap(psi, psi))
end

"""
    expect(psi, op) -> Vector{ComplexF64}

The site-resolved profile `[⟨Ô_i⟩ for i in 1:N]` of one local operator.
"""
expect(psi::MPS, op::AbstractMatrix) = ComplexF64[expect(psi, op, i) for i in 1:psi.N]

"""
    correlation(psi, op1, op2, i, j) -> ComplexF64

Normalized two-point correlation `⟨ψ|Ô¹_i Ô²_j|ψ⟩ / ⟨ψ|ψ⟩`. Operators on
distinct sites commute, so the result is independent of whether `i < j` or
`i > j`; for `i == j` the on-site product `Ô¹Ô²` is measured.
"""
function correlation(psi::MPS, op1::AbstractMatrix, op2::AbstractMatrix,
                     i::Integer, j::Integer)
    1 <= i <= psi.N || throw(BoundsError(psi, i))
    1 <= j <= psi.N || throw(BoundsError(psi, j))
    _check_op(psi, op1)
    _check_op(psi, op2)
    O1 = Array{ComplexF64}(op1)
    O2 = Array{ComplexF64}(op2)
    Osame = O1 * O2  # on-site product, used only when i == j
    E = ones(ComplexF64, 1, 1)
    for k in 1:psi.N
        insert = i == j && k == i ? Osame :
                 k == i           ? O1     :
                 k == j           ? O2     : nothing
        E = _absorb_site(E, psi.tensors[k], insert)
    end
    return E[] / real(overlap(psi, psi))
end

"""
    correlation_matrix(psi, op1, op2) -> Matrix{ComplexF64}

The full `N×N` matrix `C[i, j] = ⟨Ô¹_i Ô²_j⟩`.
"""
function correlation_matrix(psi::MPS, op1::AbstractMatrix, op2::AbstractMatrix)
    N = psi.N
    return ComplexF64[correlation(psi, op1, op2, i, j) for i in 1:N, j in 1:N]
end
