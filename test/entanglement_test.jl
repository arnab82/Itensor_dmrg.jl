using Random
using LinearAlgebra

@testset verbose = true "Entanglement entropy and Schmidt spectrum" begin
    C = NaiveDMRG

    @testset "product state has zero entanglement" begin
        up   = reshape(ComplexF64[1, 0], 1, 2, 1)
        down = reshape(ComplexF64[0, 1], 1, 2, 1)
        neel = C.MPS([copy(up), copy(down), copy(up), copy(down)])

        for b in 1:3
            @test C.schmidt_values(neel, b) ≈ [1.0] atol=1e-12
            @test C.entanglement_entropy(neel, b) ≈ 0.0 atol=1e-12
        end
        @test C.entanglement_entropy(neel) ≈ zeros(3) atol=1e-12
    end

    @testset "singlet is maximally entangled (S = log 2)" begin
        # |ψ⟩ = (|↑↓⟩ - |↓↑⟩)/√2 as a bond-dimension-2 MPS.
        A1 = zeros(ComplexF64, 1, 2, 2); A1[1, 1, 1] = 1; A1[1, 2, 2] = 1
        A2 = zeros(ComplexF64, 2, 2, 1); A2[1, 2, 1] = 1 / √2; A2[2, 1, 1] = -1 / √2
        singlet = C.MPS([A1, A2])

        @info "  singlet entanglement" schmidt = C.schmidt_values(singlet, 1) S_nats = C.entanglement_entropy(singlet, 1) S_bits = C.entanglement_entropy(singlet, 1; base=2)
        @test C.schmidt_values(singlet, 1) ≈ [1/√2, 1/√2] atol=1e-12
        @test C.entanglement_entropy(singlet, 1) ≈ log(2) atol=1e-12
        @test C.entanglement_entropy(singlet, 1; base=2) ≈ 1.0 atol=1e-12
    end

    @testset "random MPS vs dense reduced density matrix" begin
        rng = MersenneTwister(97531)
        N = 5
        psi = C.random_MPS(N, 2, 6; rng=rng)
        v = C.dense(psi)
        v ./= norm(v)
        @info "  random MPS entanglement profile (per bond)" S = C.entanglement_entropy(psi)

        for b in 1:N-1
            sref = svdvals(reshape(v, 2^b, 2^(N - b)))
            smps = C.schmidt_values(psi, b)

            @test sum(abs2, smps) ≈ 1.0 atol=1e-10
            @test sort(sref; rev=true)[1:length(smps)] ≈ smps atol=1e-9

            Sref = -sum(p * log(p) for p in sref .^ 2 if p > 1e-14)
            @test C.entanglement_entropy(psi, b) ≈ Sref atol=1e-9
        end
    end
end
