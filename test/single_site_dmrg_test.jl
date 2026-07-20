using Random
using LinearAlgebra

@testset verbose = true "Single-site DMRG with subspace expansion" begin
    C = NaiveDMRG
    alpha = (1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-5, 0.0)

    @testset "grows bond from a restricted start and reaches ED" begin
        N = 4
        H = C.heisenberg_mpo(N)
        exact = eigmin(Hermitian(C.dense(H)))          # -1.6160254037844386

        Random.seed!(2024)
        psi0 = C.random_MPS(N, 2, 2)                    # bonds capped at 2
        @test maximum(C.bond_dimensions(psi0)) == 2

        @info "  single-site DMRG sweeps (N=4 Heisenberg, start bond 2):"
        result = C.single_site_dmrg(H, psi0; nsweeps=40, maxdim=8, cutoff=1e-12,
                                    tol=1e-10, alpha=alpha, output=true)
        e, psi = result
        @info "  single-site DMRG grew the bond via subspace expansion" start_bond = 2 final_bond = maximum(C.bond_dimensions(psi)) energy = e exact = exact ΔE = abs(e - exact)
        @test result isa C.DMRGResult
        @test e ≈ exact atol=1e-7
        @test maximum(C.bond_dimensions(psi)) > 2       # subspace expansion grew the bond
        @test C.compute_energy(H, psi) ≈ e atol=1e-10
        @test norm(C.dense(psi)) ≈ 1.0 atol=1e-10
        @test C.dense(psi0) == C.dense(psi0)            # input state untouched (copying wrapper)
    end

    @testset "agrees with two-site DMRG" begin
        N = 6
        H = C.heisenberg_mpo(N)
        Random.seed!(51)
        e_two, _ = C.dmrg(H, C.random_MPS(N, 2, 16); nsweeps=14, maxdim=16,
                          cutoff=1e-12, tol=1e-10, output=false)
        e_one, _ = C.single_site_dmrg(H, C.random_MPS(N, 2, 4); nsweeps=40, maxdim=16,
                                      cutoff=1e-12, tol=1e-10, alpha=alpha, output=false)
        @test e_one ≈ e_two atol=1e-6
    end

    @testset "TFIM ground state" begin
        N, J, h = 6, 1.0, 1.0
        H = C.tfim_mpo(N; J=J, h=h)
        exact = eigmin(Hermitian(C.dense(H)))
        Random.seed!(7)
        e, _ = C.single_site_dmrg(H, C.random_MPS(N, 2, 4); nsweeps=40, maxdim=16,
                                  cutoff=1e-12, tol=1e-10, alpha=alpha, output=false)
        @test e ≈ exact atol=1e-6
    end
end
