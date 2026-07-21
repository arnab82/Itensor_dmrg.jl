# 1D Fermi-Hubbard chain in the from-scratch core (`hubbard_mpo`, d = 4).
#
# The fermion Jordan-Wigner signs are the error-prone part, so they are pinned
# numerically three ways:
#   1. the single-site operators obey fermionic anticommutation;
#   2. `dense(hubbard_mpo)` equals an INDEPENDENT many-body JW matrix (the MPO
#      automaton reproduces the operator algebra exactly);
#   3. the ground-state energy matches the ITensor "Electron" reference at half
#      filling with quantum-number conservation. Energies are invariant under
#      the JW convention, so a missing/wrong F string would show up here.

const CH = NaiveDMRG

# Independent dense Hubbard matrix from explicit many-body JW operators, in the
# little-endian (site 1 = fastest) ordering used by `NaiveDMRG.dense`. This does
# NOT use `nearest_neighbor_mpo`, so agreement with it validates the builder.
function _dense_hubbard_ref(N; t, U, mu, T=ComplexF64)
    o = CH.electron_operators(T)
    embed(site, A) = foldl(kron, reverse([k == site ? A : o.Id for k in 1:N]))
    function fermi(site, A)               # c_{site} with the ∏_{k<site} F_k string
        M = embed(site, A)
        for k in 1:site-1
            M = embed(k, o.F) * M
        end
        return M
    end
    H = zeros(T, 4^N, 4^N)
    for i in 1:N
        H .+= U .* embed(i, o.Nupdn)
        mu != 0 && (H .+= (-mu) .* (embed(i, o.Nup) + embed(i, o.Ndn)))
    end
    for i in 1:N-1, (Cd, Cc) in ((o.Cdagup, o.Cup), (o.Cdagdn, o.Cdn))
        hop = fermi(i, Cd) * fermi(i + 1, Cc)
        H .+= (-t) .* (hop + hop')
    end
    return H
end

# ITensor half-filling ground state with QN conservation (the reliable path).
function _itensor_halffilling_gs(N; t, U, mu)
    sites = siteinds("Electron", N; conserve_qns=true)
    os = OpSum()
    for i in 1:N
        os += U, "Nupdn", i
        if mu != 0
            os += -mu, "Nup", i
            os += -mu, "Ndn", i
        end
    end
    for i in 1:N-1
        os += -t, "Cdagup", i, "Cup", i + 1
        os += -t, "Cdagup", i + 1, "Cup", i
        os += -t, "Cdagdn", i, "Cdn", i + 1
        os += -t, "Cdagdn", i + 1, "Cdn", i
    end
    H = MPO(os, sites)
    # half filling: one electron per site (alternating spin), total N electrons
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = productMPS(sites, state)
    energy, _ = dmrg(H, psi0; nsweeps=14, maxdim=[20, 40, 80, 120, 200],
                     cutoff=1e-13, outputlevel=0)
    return energy
end

@testset verbose = true "1D Hubbard (from-scratch core)" begin
    @testset "electron operators are fermionic" begin
        o = CH.electron_operators(ComplexF64)
        acomm(a, b) = a * b + b * a
        @test acomm(o.Cup, o.Cdagup) ≈ o.Id           # {c,c†}=1 per mode
        @test acomm(o.Cdn, o.Cdagdn) ≈ o.Id
        @test maximum(abs, acomm(o.Cup, o.Cup)) < 1e-14   # {c,c}=0
        @test maximum(abs, acomm(o.Cdn, o.Cdn)) < 1e-14
        @test maximum(abs, acomm(o.Cup, o.Cdn)) < 1e-14   # different spins anticommute
        @test maximum(abs, acomm(o.Cup, o.Cdagdn)) < 1e-14
        @test real(diag(o.Nup)) == [0, 1, 0, 1]
        @test real(diag(o.Ndn)) == [0, 0, 1, 1]
        @test real(diag(o.Nupdn)) == [0, 0, 0, 1]
        @test real(diag(o.F)) == [1, -1, -1, 1]
        @test o.F ≈ o.Id - 2o.Nup - 2o.Ndn + 4 * (o.Nup * o.Ndn)  # F = (-1)^(n↑+n↓)
    end

    @testset "hubbard_mpo == independent JW matrix (exact)" begin
        for (N, t, U, mu) in [(2, 1.0, 4.0, 0.0), (4, 1.3, 5.0, 2.5), (5, 1.0, 8.0, 0.0)]
            Hd = CH.dense(CH.hubbard_mpo(N; t=t, U=U, mu=mu))
            @test Hd ≈ Hd' atol=1e-12                             # Hermitian
            @test Hd ≈ _dense_hubbard_ref(N; t=t, U=U, mu=mu) atol=1e-12
            # bond dimension w = K + 2 = 6 (four hopping channels)
            @test size(CH.hubbard_mpo(N; t=t, U=U, mu=mu).tensors[2], 1) == 6
        end
    end

    @testset "custom DMRG reaches the exact ground state" begin
        for (N, t, U, mu) in [(4, 1.0, 4.0, 2.0), (6, 1.0, 8.0, 4.0)]
            H = CH.hubbard_mpo(N; t=t, U=U, mu=mu)
            exact = eigmin(Hermitian(CH.dense(H)))
            e, _ = CH.dmrg(H, CH.random_MPS(N, 4, 32); nsweeps=18, maxdim=64,
                           cutoff=1e-12, tol=1e-11, output=false)
            @info "  Hubbard ground state (custom)" N=N U=U mu=mu dmrg=e exact=exact
            @test e ≈ exact atol=1e-8
        end
    end

    @testset "ground state matches ITensor Electron reference" begin
        # particle-hole symmetric point mu = U/2: half filling is the global
        # ground state, so the from-scratch (all-sector) energy equals ITensor's
        # QN-conserved half-filling energy.
        for (N, t, U) in [(4, 1.0, 4.0), (4, 1.0, 8.0), (6, 1.0, 4.0)]
            mu = U / 2
            H = CH.hubbard_mpo(N; t=t, U=U, mu=mu, T=Float64)
            e, _ = CH.dmrg(H, CH.random_MPS(N, 4, 32; T=Float64); nsweeps=18,
                           maxdim=80, cutoff=1e-12, tol=1e-11, output=false)
            e_ref = _itensor_halffilling_gs(N; t=t, U=U, mu=mu)
            @info "  Hubbard vs ITensor (half filling, mu=U/2)" N=N U=U custom=e itensor=e_ref
            @test e ≈ e_ref atol=1e-6
        end
    end
end
