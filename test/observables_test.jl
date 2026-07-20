using Random
using LinearAlgebra

@testset verbose = true "Local observables and correlations" begin
    C = NaiveDMRG
    ops = C.spin_half_operators()
    Sz, Sp, Sm, Id = ops.Sz, ops.Sp, ops.Sm, ops.Id

    # Dense reference operator: site 1 is the least-significant (fastest) index,
    # matching the reshape order used by NaiveDMRG.dense.
    dense_op(N, placements) =
        foldl(kron, reverse([get(placements, k, Id) for k in 1:N]))

    @testset "product (Néel) state — exact by construction" begin
        up   = reshape(ComplexF64[1, 0], 1, 2, 1)
        down = reshape(ComplexF64[0, 1], 1, 2, 1)
        neel = C.MPS([copy(up), copy(down), copy(up), copy(down)])

        @test C.expect(neel, Sz, 1) ≈  0.5 atol=1e-12
        @test C.expect(neel, Sz, 2) ≈ -0.5 atol=1e-12
        @test real.(C.expect(neel, Sz)) ≈ [0.5, -0.5, 0.5, -0.5] atol=1e-12

        @test C.correlation(neel, Sz, Sz, 1, 2) ≈ -0.25 atol=1e-12
        @test C.correlation(neel, Sz, Sz, 1, 3) ≈  0.25 atol=1e-12
        @test C.correlation(neel, Sz, Sz, 2, 2) ≈  0.25 atol=1e-12   # on-site Sz² = 1/4
        @test C.correlation(neel, Sp, Sm, 1, 2) ≈  0.0  atol=1e-12   # S⁻ on |↓⟩ = 0
    end

    @testset "random MPS vs dense reference" begin
        rng = MersenneTwister(4242)
        N = 5
        psi = C.random_MPS(N, 2, 6; rng=rng)
        v = C.dense(psi)
        nv = real(dot(v, v))

        for i in 1:N
            ref = dot(v, dense_op(N, Dict(i => Sz)) * v) / nv
            @test C.expect(psi, Sz, i) ≈ ref atol=1e-10
        end

        for i in 1:N, j in 1:N
            ref = if i == j
                dot(v, dense_op(N, Dict(i => Sz * Sz)) * v) / nv
            else
                dot(v, dense_op(N, Dict(i => Sz, j => Sz)) * v) / nv
            end
            @test C.correlation(psi, Sz, Sz, i, j) ≈ ref atol=1e-10
        end

        # Non-Hermitian operators must also match (complex-valued results).
        for i in 1:N, j in 1:N
            i == j && continue
            ref = dot(v, dense_op(N, Dict(i => Sp, j => Sm)) * v) / nv
            @test C.correlation(psi, Sp, Sm, i, j) ≈ ref atol=1e-10
        end
    end

    @testset "Heisenberg ground state — physical checks" begin
        Random.seed!(8128)
        H = C.heisenberg_mpo(4)
        _, gs = C.dmrg(H, C.random_MPS(4, 2, 8); nsweeps=8, maxdim=8,
                       cutoff=1e-12, tol=1e-10, output=false)

        @test abs(sum(real, C.expect(gs, Sz))) < 1e-7        # total Sz = 0 sector
        Cm = C.correlation_matrix(gs, Sz, Sz)
        @info "  Heisenberg GS observables (N=4)" total_Sz = sum(real, C.expect(gs, Sz)) Sz1_Sz2 = real(Cm[1, 2]) Sz1_sq = real(Cm[1, 1])
        @test Cm ≈ Cm' atol=1e-9                             # ⟨SᶻᵢSᶻⱼ⟩ = ⟨SᶻⱼSᶻᵢ⟩
        @test all(abs(Cm[i, i] - 0.25) < 1e-9 for i in 1:4)  # ⟨Sᶻᵢ²⟩ = 1/4
        @test abs(real(sum(Cm))) < 1e-6                      # ⟨(ΣSᶻ)²⟩ = 0
    end
end
