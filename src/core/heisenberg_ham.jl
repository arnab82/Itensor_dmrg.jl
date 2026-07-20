"""
    heisenberg_mpo(N; J=1.0, hz=0.0, T=ComplexF64)

Open-chain spin-1/2 Hamiltonian
`J * sum(SxₙSxₙ₊₁ + SyₙSyₙ₊₁ + SzₙSzₙ₊₁) + hz * sum(Szₙ)`.
The model is real, so `T=Float64` gives a real MPO (and a fully real solve).
"""
function heisenberg_mpo(N::Integer; J::Real=1.0, hz::Real=0.0,
                        T::Type{<:Number}=ComplexF64)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    id = T[1 0; 0 1]
    sp = T[0 1; 0 0]
    sm = T[0 0; 1 0]
    sz = T[0.5 0; 0 -0.5]
    onsite = hz .* sz

    # Finite-state MPO: completed term, three open interactions, identity path.
    bulk = zeros(T, 5, 2, 2, 5)
    bulk[1, :, :, 1] .= id
    bulk[2, :, :, 1] .= sm
    bulk[3, :, :, 1] .= sp
    bulk[4, :, :, 1] .= sz
    bulk[5, :, :, 1] .= onsite
    bulk[5, :, :, 2] .= (J / 2) .* sp
    bulk[5, :, :, 3] .= (J / 2) .* sm
    bulk[5, :, :, 4] .= J .* sz
    bulk[5, :, :, 5] .= id

    tensors = Vector{Array{T,4}}(undef, N)
    tensors[1] = copy(bulk[5:5, :, :, :])
    tensors[2:N-1] .= Ref(copy(bulk))
    tensors[N] = copy(bulk[:, :, :, 1:1])
    return MPO(tensors)
end

# Compatibility with the former API; its MPO bond dimension is necessarily five.
function heisenberg_ham(N::Integer, d::Integer=2, chi::Integer=5)
    d == 2 || throw(ArgumentError("the spin-1/2 model requires d=2"))
    chi == 5 || throw(ArgumentError("the Heisenberg MPO has bond dimension 5"))
    return heisenberg_mpo(N)
end
