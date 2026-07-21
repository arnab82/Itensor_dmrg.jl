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
# `bonds` are chain-index (p, q) pairs; for a chain use consecutive pairs.
function _itensor_halffilling_gs(N, bonds; t, U, mu)
    sites = siteinds("Electron", N; conserve_qns=true)
    os = OpSum()
    for i in 1:N
        os += U, "Nupdn", i
        if mu != 0
            os += -mu, "Nup", i
            os += -mu, "Ndn", i
        end
    end
    for (p, q) in bonds
        os += -t, "Cdagup", p, "Cup", q
        os += -t, "Cdagup", q, "Cup", p
        os += -t, "Cdagdn", p, "Cdn", q
        os += -t, "Cdagdn", q, "Cdn", p
    end
    H = MPO(os, sites)
    # half filling: one electron per site (alternating spin), total N electrons
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = productMPS(sites, state)
    energy, _ = dmrg(H, psi0; nsweeps=16, maxdim=[20, 40, 100, 200, 300],
                     cutoff=1e-13, outputlevel=0)
    return energy
end

_chain_bonds(N) = [(i, i + 1) for i in 1:N-1]

function _lattice_bonds(Nx, Ny)
    site(i, j) = i + (j - 1) * Nx
    bonds = Tuple{Int,Int}[]
    for j in 1:Ny, i in 1:Nx
        i < Nx && push!(bonds, (site(i, j), site(i + 1, j)))
        j < Ny && push!(bonds, (site(i, j), site(i, j + 1)))
    end
    return bonds
end

# Independent dense realization of a general_mpo term list (little-endian),
# not routed through general_mpo — validates the FSM compiler. The factors of a
# two-site+string term act on DISJOINT sites, so the term is a single kron
# placement (no matmuls), keeping this fast at N=6.
function _dense_from_terms(N; onsite, twosite, T=ComplexF64)
    o = CH.electron_operators(T)
    place(factors) = foldl(kron, reverse([get(factors, k, o.Id) for k in 1:N]))
    H = zeros(T, 4^N, 4^N)
    for (c, s, A) in onsite
        H .+= convert(T, c) .* place(Dict(s => A))
    end
    for (c, sL, L, sR, R, S) in twosite
        f = Dict(sL => L, sR => R)
        for m in (sL+1):(sR-1)
            f[m] = S
        end
        H .+= convert(T, c) .* place(f)
    end
    return H
end

# Reproduce the exact term list hubbard_2d_mpo feeds general_mpo.
function _hub2d_terms(Nx, Ny; t, U, mu)
    o = CH.electron_operators(ComplexF64)
    site(i, j) = i + (j - 1) * Nx
    onsite = Tuple[]
    twosite = Tuple[]
    for j in 1:Ny, i in 1:Nx
        p = site(i, j)
        push!(onsite, (U, p, o.Nupdn))
        if mu != 0
            push!(onsite, (-mu, p, o.Nup))
            push!(onsite, (-mu, p, o.Ndn))
        end
    end
    hop(p, q) = [(-t, p, o.Cdagup * o.F, q, o.Cup, o.F),
                 (-t, p, o.F * o.Cup, q, o.Cdagup, o.F),
                 (-t, p, o.Cdagdn * o.F, q, o.Cdn, o.F),
                 (-t, p, o.F * o.Cdn, q, o.Cdagdn, o.F)]
    for (p, q) in _lattice_bonds(Nx, Ny)
        append!(twosite, hop(p, q))
    end
    return onsite, twosite
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
            e_ref = _itensor_halffilling_gs(N, _chain_bonds(N); t=t, U=U, mu=mu)
            @info "  Hubbard vs ITensor (half filling, mu=U/2)" N=N U=U custom=e itensor=e_ref
            @test e ≈ e_ref atol=1e-6
        end
    end

    @testset "general_mpo long-range compiler" begin
        # bosonic long-range ZZ (string = I) vs an independent dense build
        Sz = ComplexF64[0.5 0; 0 -0.5]
        Id2 = ComplexF64[1 0; 0 1]
        N = 6
        twosite = [(0.7, 1, Sz, 4, Sz, Id2), (0.3, 2, Sz, 6, Sz, Id2)]
        onsite = [(0.5, 3, Sz)]
        H = CH.general_mpo(N, 2; onsite=onsite, twosite=twosite)
        emb(s, A) = foldl(kron, reverse([k == s ? A : Id2 for k in 1:N]))
        Href2 = 0.5 * emb(3, Sz) + 0.7 * emb(1, Sz) * emb(2, Id2) * emb(3, Id2) * emb(4, Sz) +
                0.3 * emb(2, Sz) * emb(6, Sz)
        @test CH.dense(H) ≈ Href2 atol=1e-12
        @test CH.dense(H) ≈ CH.dense(H)' atol=1e-12
    end

    @testset "hubbard_2d_mpo reduces to 1D at Ny=1" begin
        for Nx in (2, 4, 5)
            @test CH.dense(CH.hubbard_2d_mpo(Nx, 1; t=1.0, U=3.0, mu=1.5)) ≈
                  CH.dense(CH.hubbard_mpo(Nx; t=1.0, U=3.0, mu=1.5)) atol=1e-12
        end
    end

    @testset "hubbard_2d_mpo == independent JW matrix (exact)" begin
        for (Nx, Ny, U, mu) in [(2, 2, 4.0, 2.0), (2, 3, 5.0, 0.0), (3, 2, 4.0, 2.0)]
            H = CH.hubbard_2d_mpo(Nx, Ny; t=1.0, U=U, mu=mu)
            Hd = CH.dense(H)
            on, tw = _hub2d_terms(Nx, Ny; t=1.0, U=U, mu=mu)
            @test Hd ≈ Hd' atol=1e-12
            @test Hd ≈ _dense_from_terms(Nx * Ny; onsite=on, twosite=tw) atol=1e-12
        end
    end

    @testset "2D ground state matches ITensor Electron reference" begin
        for (Nx, Ny, U) in [(2, 2, 4.0), (2, 3, 4.0), (3, 2, 8.0)]
            mu = U / 2
            H = CH.hubbard_2d_mpo(Nx, Ny; t=1.0, U=U, mu=mu, T=Float64)
            e, _ = CH.dmrg(H, CH.random_MPS(Nx * Ny, 4, 48; T=Float64); nsweeps=20,
                           maxdim=120, cutoff=1e-12, tol=1e-11, output=false)
            e_ref = _itensor_halffilling_gs(Nx * Ny, _lattice_bonds(Nx, Ny); t=1.0, U=U, mu=mu)
            @info "  2D Hubbard vs ITensor (half filling)" Nx=Nx Ny=Ny U=U custom=e itensor=e_ref
            @test e ≈ e_ref atol=1e-6
        end
    end
end
