using Random

@testset "NaiveDMRG two-site DMRG" begin
    C = NaiveDMRG
    N = 4
    Random.seed!(8128)

    H_custom = C.heisenberg_mpo(N)
    H_dense = C.dense(H_custom)
    exact_energy = eigmin(Hermitian(H_dense))

    @test H_dense ≈ H_dense'
    @test exact_energy ≈ -1.6160254037844386 atol=1e-12

    psi0 = C.random_MPS(N, 2, 8)
    initial_vector = C.dense(psi0)
    result = C.dmrg(H_custom, psi0; nsweeps=8, maxdim=8,
                    cutoff=1e-12, tol=1e-10, output=false)
    custom_energy, psi = result

    @test result isa C.DMRGResult
    @test result.converged
    @test result.stopping_reason == :energy_tolerance
    @test !isempty(result.history)
    @test all(length(s.local_solves) == 2 * (N - 1) for s in result.history)
    @test all(solve.converged for sweep in result.history for solve in sweep.local_solves)
    @test all(solve.residual_norm <= 1e-9 for sweep in result.history for solve in sweep.local_solves)
    @test C.dense(psi0) == initial_vector
    @test result.state !== psi0
    @test custom_energy ≈ exact_energy atol=1e-9
    @test C.compute_energy(H_custom, psi) ≈ custom_energy atol=1e-11
    @test norm(C.dense(psi)) ≈ 1.0 atol=1e-11
    @test maximum(C.bond_dimensions(psi)) <= 8

    mutating_state = copy(psi0)
    mutating_result = C.dmrg!(H_custom, mutating_state; nsweeps=2, maxdim=8,
                              cutoff=1e-12, tol=1e-10, output=false)
    @test mutating_result.state === mutating_state
    @test C.dense(mutating_state) != initial_vector

    scheduled = C.dmrg(
        H_custom,
        psi0;
        nsweeps=2,
        maxdim=(2, 8),
        cutoff=(1e-6, 1e-12),
        tol=(0.0, 1e-10),
        eig_tol=(1e-8, 1e-10),
        output=false,
    )
    @test [s.maxdim_limit for s in scheduled.history] == [2, 8]
    @test [s.cutoff for s in scheduled.history] == [1e-6, 1e-12]
    @test [s.tolerance for s in scheduled.history] == [0.0, 1e-10]
    @test scheduled.history[1].max_bond_dimension <= 2
    @test scheduled.history[2].max_bond_dimension <= 8

    # ITensor is a reference calculation only; the naive path above does not call it.
    sites = siteinds("S=1/2", N)
    H_itensor = ITensorMPS.MPO(NaiveDMRG.Reference.heisenberg_hamiltonian(N, 1, 1.0), sites)
    initial = productMPS(sites, [isodd(i) ? "Up" : "Dn" for i in 1:N])
    itensor_energy, _ = ITensorMPS.dmrg(H_itensor, initial;
                                        nsweeps=8, maxdim=8, cutoff=1e-12,
                                        outputlevel=0)
    @info "  two-site DMRG ground state (N=4 Heisenberg)" exact = exact_energy custom = custom_energy itensor = itensor_energy
    @test custom_energy ≈ itensor_energy atol=1e-9
end
