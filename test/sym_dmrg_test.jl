# Stage 3: block-sparse two-site DMRG (symmetry=true).
#
# Two independent pins against the (ITensor-validated) dense path:
#   1. the symmetric environment contraction reproduces ⟨ψ|H|ψ⟩ of the dense
#      state, so the block-sparse MPS–MPO–MPS network is wired correctly;
#   2. the symmetric ground-state energy equals exact diagonalization at half
#      filling / Sz=0, so the full sweep (effective action, SVD split,
#      canonicalization) is correct.

const CH = NaiveDMRG
const SQN = NaiveDMRG.QN

# ⟨ψ|H|ψ⟩ by folding the symmetric left environment across the whole chain.
function sym_energy(H, psi)
    T = promote_type(eltype(H), eltype(psi))
    L = CH.left_boundary(T, Val(length(psi.sector)))
    for i in 1:psi.N
        L = CH.absorb_left_sym(L, psi.tensors[i], H.tensors[i])
    end
    return real(only(values(L.blocks))[1])
end

@testset verbose = true "Symmetric DMRG (Stage 3)" begin
    @testset "environment energy matches the dense contraction" begin
        for (N, U, mu) in [(3, 4.0, 2.0), (4, 8.0, 4.0)]
            H = CH.hubbard_mpo(N; U=U, mu=mu, T=Float64, symmetry=true)
            psi = CH.random_sym_mps([CH.electron_site_qns() for _ in 1:N],
                                    CH.electron_half_filling(N), 6; T=Float64, perbond=2)
            CH.right_canonicalize_sym!(psi)
            CH.normalize_sym!(psi)
            @test sym_energy(H, psi) ≈ CH.compute_energy(CH.dense(H), CH.dense(psi)) atol = 1e-9
        end
    end

    @testset "Hubbard ground-state energy equals exact diagonalization" begin
        for (N, U) in [(4, 8.0), (6, 8.0)]
            mu = U / 2
            H = CH.hubbard_mpo(N; U=U, mu=mu, T=Float64, symmetry=true)
            psi0 = CH.random_MPS(H, 40; sector=CH.electron_half_filling(N), T=Float64, perbond=2)
            e, _ = CH.dmrg(H, psi0; nsweeps=24, maxdim=60, cutoff=1e-11, tol=1e-11, output=false)
            exact = eigmin(Hermitian(CH.dense(CH.hubbard_mpo(N; U=U, mu=mu, T=Float64))))
            @info "  symmetric Hubbard GS" N=N U=U sym=e exact=exact
            @test e ≈ exact atol = 1e-7
        end
    end

    @testset "Heisenberg ground-state energy equals exact diagonalization" begin
        N = 6
        H = CH.heisenberg_mpo(N; J=1.0, T=Float64, symmetry=true)
        psi0 = CH.random_MPS(H, 40; sector=SQN(0), T=Float64, perbond=2)
        e, _ = CH.dmrg(H, psi0; nsweeps=24, maxdim=60, cutoff=1e-11, tol=1e-11, output=false)
        exact = eigmin(Hermitian(CH.dense(CH.heisenberg_mpo(N; J=1.0, T=Float64))))
        @info "  symmetric Heisenberg GS" N=N sym=e exact=exact
        @test e ≈ exact atol = 1e-7
    end
end
