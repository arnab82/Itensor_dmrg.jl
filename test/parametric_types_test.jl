using Random
using LinearAlgebra

@testset verbose = true "Parametric scalar types" begin
    C = NaiveDMRG
    N = 6

    @testset "real (Float64) types propagate" begin
        Hr = C.heisenberg_mpo(N; T=Float64)
        @test Hr isa C.MPO{Float64}
        @test eltype(Hr) == Float64
        psir = C.random_MPS(N, 2, 8; rng=MersenneTwister(1), T=Float64)
        @test psir isa C.MPS{Float64}
        @test eltype(psir) == Float64
        @test eltype(C.dense(Hr)) == Float64
        @test eltype(C.dense(psir)) == Float64
        @test C.dense(Hr) ≈ C.dense(C.heisenberg_mpo(N))   # same numbers as the complex MPO
    end

    @testset "full real DMRG matches exact diagonalization" begin
        Hr = C.heisenberg_mpo(N; T=Float64)
        exact = eigmin(Hermitian(C.dense(Hr)))
        e, psi = C.dmrg(Hr, C.random_MPS(N, 2, 16; rng=MersenneTwister(2), T=Float64);
                        nsweeps=12, maxdim=16, cutoff=1e-12, tol=1e-10, output=false)
        @info "  full real (Float64) DMRG" energy = e exact = exact eltype = eltype(psi)
        @test psi isa C.MPS{Float64}
        @test e isa Float64
        @test e ≈ exact atol=1e-9

        es, _ = C.single_site_dmrg(Hr, C.random_MPS(N, 2, 4; rng=MersenneTwister(3), T=Float64);
                                   nsweeps=30, maxdim=16, cutoff=1e-12, tol=1e-10,
                                   alpha=(1e-2, 1e-3, 1e-4, 0.0), output=false)
        @test es isa Float64
        @test es ≈ exact atol=1e-7
    end

    @testset "type promotion between H and psi" begin
        Hc = C.heisenberg_mpo(N)                                     # ComplexF64
        realpsi = C.random_MPS(N, 2, 16; rng=MersenneTwister(4), T=Float64)
        # The copying `dmrg` promotes the real state to match the complex H.
        _, psi = C.dmrg(Hc, realpsi; nsweeps=10, maxdim=16, cutoff=1e-12, tol=1e-10, output=false)
        @test psi isa C.MPS{ComplexF64}
        @test realpsi isa C.MPS{Float64}                            # input untouched
        # In-place `dmrg!` cannot retype the state → an informative error before mutating.
        @test_throws ArgumentError C.dmrg!(Hc, realpsi; nsweeps=1, output=false)
        @test realpsi isa C.MPS{Float64}
    end

    @testset "builders and observables honor T" begin
        Sz = Float64[0.5 0; 0 -0.5]
        Hb = C.nearest_neighbor_mpo(N, 2; onsite=[(0.3, Sz)], bond=[(1.0, Sz, Sz)], T=Float64)
        @test Hb isa C.MPO{Float64}
        @test C.tfim_mpo(N; T=Float64) isa C.MPO{Float64}

        psir = C.random_MPS(N, 2, 8; rng=MersenneTwister(5), T=Float64)
        @test eltype(C.expect(psir, Sz)) == Float64
        @test C.expect(psir, Sz, 1) isa Float64
        @test eltype(C.correlation_matrix(psir, Sz, Sz)) == Float64
        @test eltype(C.entanglement_entropy(psir)) == Float64
        # A complex operator on a real state promotes to complex.
        @test C.expect(psir, C.spin_half_operators().Sy, 1) isa Complex
    end
end
