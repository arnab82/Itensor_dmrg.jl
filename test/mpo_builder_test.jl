using LinearAlgebra

@testset verbose = true "Generic nearest-neighbor MPO builder" begin
    C = NaiveDMRG
    Sx = ComplexF64[0 0.5; 0.5 0]
    Sz = ComplexF64[0.5 0; 0 -0.5]
    Sp = ComplexF64[0 1; 0 0]
    Sm = ComplexF64[0 0; 1 0]
    Id = ComplexF64[1 0; 0 1]

    # Dense operator with site 1 as the least-significant (fastest) index,
    # matching NaiveDMRG.dense.
    dense_op(N, placements) =
        foldl(kron, reverse([get(placements, k, Id) for k in 1:N]))

    @testset "reproduces heisenberg_mpo" begin
        for (N, J, hz) in [(4, 1.0, 0.0), (5, 0.7, 0.3)]
            built = C.nearest_neighbor_mpo(N, 2;
                onsite = [(hz, Sz)],
                bond   = [(J / 2, Sp, Sm), (J / 2, Sm, Sp), (J, Sz, Sz)])
            Δ = maximum(abs, C.dense(built) - C.dense(C.heisenberg_mpo(N; J=J, hz=hz)))
            @info "  builder reproduces heisenberg_mpo" N = N J = J hz = hz max_abs_diff = Δ
            @test C.dense(built) ≈ C.dense(C.heisenberg_mpo(N; J=J, hz=hz)) atol=1e-12
            @test size(built.tensors[2], 1) == 5          # w = K + 2 = 5
            @test size(built.tensors[2], 4) == 5
        end
    end

    @testset "TFIM vs dense and exact diagonalization" begin
        N, J, h = 6, 1.0, 1.0
        H = C.tfim_mpo(N; J=J, h=h)

        Hdense = zeros(ComplexF64, 2^N, 2^N)
        for i in 1:N
            Hdense .+= -h .* dense_op(N, Dict(i => Sx))
        end
        for i in 1:N-1
            Hdense .+= -J .* dense_op(N, Dict(i => Sz, i + 1 => Sz))
        end

        @test C.dense(H) ≈ Hdense atol=1e-12
        @test C.dense(H) ≈ C.dense(H)' atol=1e-12          # Hermitian

        exact = eigmin(Hermitian(Hdense))
        e, _ = C.dmrg(H, C.random_MPS(N, 2, 16); nsweeps=12, maxdim=16,
                      cutoff=1e-12, tol=1e-10, output=false)
        @info "  TFIM(N=6, J=h=1) ground state" dmrg = e exact = exact
        @test e ≈ exact atol=1e-6
    end

    @testset "on-site only (bond dimension 2)" begin
        N = 4
        # H = Σ Sᶻᵢ : diagonal, ground state all-down with energy -N/2.
        H = C.nearest_neighbor_mpo(N, 2; onsite = [(1.0, Sz)])
        @test size(H.tensors[2], 1) == 2
        @test eigmin(Hermitian(C.dense(H))) ≈ -N / 2 atol=1e-12
    end
end
