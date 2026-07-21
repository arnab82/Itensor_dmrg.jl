# Stage 2: symmetric MPO/MPS builders for the symmetry=true path.
#
# The dense path is already validated against ITensor, so the symmetric builders
# are pinned to it: `dense(symmetrize(H)) == dense(H)` for Hubbard (1D + 2D) and
# Heisenberg, and a random symmetric MPS is a valid dense state whose measured
# charges equal its target sector.

const CH = NaiveDMRG
const SQN = NaiveDMRG.QN

@testset verbose = true "Symmetric builders (Stage 2)" begin
    @testset "operator charges are detected" begin
        o = CH.electron_operators(Float64)
        sq = CH.electron_site_qns()
        @test CH.op_charge(o.Cdagup, sq) == SQN(1, 0)
        @test CH.op_charge(o.Cup, sq) == SQN(-1, 0)
        @test CH.op_charge(o.Cdagdn, sq) == SQN(0, 1)
        @test CH.op_charge(o.Cdn, sq) == SQN(0, -1)
        @test CH.op_charge(o.F, sq) == SQN(0, 0)
        @test CH.op_charge(o.Nupdn, sq) == SQN(0, 0)
        @test CH.op_charge(o.Cdagup * o.F, sq) == SQN(1, 0)     # JW-dressed hop
        # spin-1/2: S± shift 2·Sz by ±2
        sp = CH.spin_half_operators()
        ssq = CH.spinhalf_site_qns()
        @test CH.op_charge(sp.Sp, ssq) == SQN(2)
        @test CH.op_charge(sp.Sm, ssq) == SQN(-2)
    end

    @testset "symmetric Hubbard MPO == dense (1D)" begin
        for (N, t, U, mu) in [(2, 1.0, 4.0, 0.0), (4, 1.3, 5.0, 2.5), (5, 1.0, 8.0, 4.0)]
            Hsym = CH.hubbard_mpo(N; t=t, U=U, mu=mu, T=Float64, symmetry=true)
            @test Hsym isa CH.SymMPO
            @test CH.dense(CH.dense(Hsym)) ≈ CH.dense(CH.hubbard_mpo(N; t=t, U=U, mu=mu, T=Float64))
        end
    end

    @testset "symmetric Hubbard MPO == dense (2D)" begin
        for (Nx, Ny, U, mu) in [(2, 2, 4.0, 2.0), (2, 3, 5.0, 0.0), (3, 2, 8.0, 4.0)]
            Hsym = CH.hubbard_2d_mpo(Nx, Ny; U=U, mu=mu, T=Float64, symmetry=true)
            @test CH.dense(CH.dense(Hsym)) ≈
                  CH.dense(CH.hubbard_2d_mpo(Nx, Ny; U=U, mu=mu, T=Float64))
        end
    end

    @testset "symmetric Heisenberg MPO == dense; TFIM refuses" begin
        for (N, J, hz) in [(4, 1.0, 0.0), (5, 1.234, 0.3)]
            Hsym = CH.heisenberg_mpo(N; J=J, hz=hz, T=Float64, symmetry=true)
            @test CH.dense(CH.dense(Hsym)) ≈ CH.dense(CH.heisenberg_mpo(N; J=J, hz=hz, T=Float64))
        end
        # the transverse field breaks Sz conservation, so symmetrization must fail
        @test_throws ArgumentError CH.symmetrize_mpo(CH.tfim_mpo(4; T=Float64),
                                                     [CH.spinhalf_site_qns() for _ in 1:4])
    end

    @testset "random symmetric MPS sits in its charge sector" begin
        N = 6
        sec = CH.electron_half_filling(N)                 # (3, 3)
        sites = [CH.electron_site_qns() for _ in 1:N]
        psi = CH.random_sym_mps(sites, sec, 8; T=Float64, perbond=2)
        @test psi isa CH.SymMPS
        psid = CH.dense(psi)                              # a plain dense MPS
        @test CH.norm(psid) > 0
        o = CH.electron_operators(Float64)
        nup = sum(real(CH.expect(psid, o.Nup, i)) for i in 1:N)
        ndn = sum(real(CH.expect(psid, o.Ndn, i)) for i in 1:N)
        @test nup ≈ sec.q[1] atol = 1e-9
        @test ndn ≈ sec.q[2] atol = 1e-9

        # spin-1/2 Sz = 0 sector
        ssites = [CH.spinhalf_site_qns() for _ in 1:N]
        spsi = CH.random_sym_mps(ssites, SQN(0), 8; T=Float64, perbond=2)
        sp = CH.spin_half_operators()
        sztot = sum(real(CH.expect(CH.dense(spsi), sp.Sz, i)) for i in 1:N)
        @test sztot ≈ 0 atol = 1e-9
    end

    @testset "random_MPS seeded from a SymMPO" begin
        N = 4
        H = CH.hubbard_mpo(N; U=8.0, mu=4.0, T=Float64, symmetry=true)
        psi = CH.random_MPS(H, 8; sector=CH.electron_half_filling(N), T=Float64, perbond=2)
        @test psi isa CH.SymMPS
        @test length(CH.bond_dimensions(psi)) == N - 1
    end
end
