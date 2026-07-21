# Fermi-Hubbard DMRG with the optional U(1) block-sparse path (`symmetry=true`).
#
# Unlike the dense solver (example/hubbard.jl), which finds only the GLOBAL ground
# state over all fillings — forcing the particle-hole trick mu = U/2 to land on
# half filling — the symmetric solver targets a SPECIFIC (N up, N dn) charge
# sector directly. So we can ask for the ground state at half filling, or at any
# other filling, with mu = 0 and no tricks.
#
# Note on performance: for these small d=4 chains the block-sparse path is a bit
# SLOWER than the tuned dense path (the charge blocks are tiny, so per-block
# overhead dominates). Its value here is the ability to fix the charge sector and
# read off exact quantum numbers, not speed.
#
# Run:  julia --project=. example/hubbard_symmetry.jl

using NaiveDMRG
using Printf
using Random

rng = MersenneTwister(2024)

function run_sector(N, t, U, sector; label)
    # mu = 0: within a fixed particle-number sector the chemical potential is a
    # constant energy shift, so it does not affect the state.
    H = hubbard_mpo(N; t=t, U=U, mu=0.0, T=Float64, symmetry=true)
    psi0 = random_MPS(H, 16; sector=sector, T=Float64, rng=rng, perbond=2)
    energy, psi = dmrg(H, psi0; nsweeps=20, maxdim=[16, 32, 64],
                       cutoff=1e-11, tol=1e-10, output=false)

    ops = electron_operators(Float64)
    nup = sum(real(expect(psi, ops.Nup, i)) for i in 1:N)
    ndn = sum(real(expect(psi, ops.Ndn, i)) for i in 1:N)
    docc = sum(real(expect(psi, ops.Nupdn, i)) for i in 1:N) / N

    @printf("\n%s  N=%d  t=%.1f  U=%.1f\n", label, N, t, U)
    @printf("  target sector (N up, N dn) : (%d, %d)\n", sector.q[1], sector.q[2])
    @printf("  ground-state energy        : %.10f\n", energy)
    @printf("  measured (N up, N dn)      : (%.4f, %.4f)\n", nup, ndn)
    @printf("  total electrons            : %.4f\n", nup + ndn)
    @printf("  mean double occupancy      : %.6f\n", docc)
    return energy
end

# ---------------------------------------------------------------------------
# 1D Hubbard chain, targeting different fillings of the SAME Hamiltonian.
# ---------------------------------------------------------------------------
N, t, U = 8, 1.0, 8.0

# Half filling: 4 up + 4 down = 8 electrons on 8 sites.
run_sector(N, t, U, electron_half_filling(N); label="Half filling")

# Two holes away from half filling: 3 up + 3 down = 6 electrons. The dense solver
# cannot isolate this; here it is just a different target sector.
run_sector(N, t, U, QN(3, 3); label="Two holes (6 electrons)")

# A spin-imbalanced sector: 5 up + 3 down (magnetized).
run_sector(N, t, U, QN(5, 3); label="Spin-imbalanced (5 up, 3 dn)")
