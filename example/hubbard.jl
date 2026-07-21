# Fermi-Hubbard DMRG with the from-scratch solver (no ITensor).
#
# The from-scratch core carries no particle-number symmetry, so the solver
# finds the GLOBAL ground state over all fillings. Choosing the chemical
# potential mu = U/2 puts the model at the particle-hole-symmetric point, where
# half filling is the global minimum — so these runs return half-filling
# ground states directly.
#
# NOTE ON SIZE: this is a DENSE d=4 DMRG with no U(1)/Sz block-sparsity. Beyond
# the two independent limits (no symmetry blocking; exponential dense fallback),
# the naive hot loop currently allocates heavily at d=4 — a ~6-site solve churns
# tens of GB of temporaries — so it is only practical for small systems today.
# The demos below are deliberately tiny so they run in a few seconds; a
# symmetric, buffer-reusing solver (e.g. ITensor) is the right tool at scale.
# See docs/src/roadmap.md for the planned fixes.
#
# Run:  julia --project=. example/hubbard.jl

using NaiveDMRG
using Printf
using Random

rng = MersenneTwister(1234)

# ---------------------------------------------------------------------------
# 1D Hubbard chain: H = -t Σ_σ (c†_iσ c_{i+1,σ} + h.c.) + U Σ n↑n↓ - μ Σ n
# ---------------------------------------------------------------------------
let
    N, t, U = 6, 1.0, 8.0
    mu = U / 2                                   # half filling (p-h symmetric)
    H = hubbard_mpo(N; t=t, U=U, mu=mu, T=Float64)
    psi0 = random_MPS(N, 4, 16; T=Float64, rng=rng)

    energy, psi = dmrg(H, psi0; nsweeps=12,
                       maxdim=[16, 24, 32],
                       cutoff=1e-10, tol=1e-8)

    # occupations should be ~1 electron/site at half filling
    ops = electron_operators(Float64)
    n_site = [real(expect(psi, ops.Nup + ops.Ndn, i)) for i in 1:N]
    docc = [real(expect(psi, ops.Nupdn, i)) for i in 1:N]

    @printf("\n1D Hubbard chain  N=%d  t=%.1f  U=%.1f  (mu=U/2)\n", N, t, U)
    @printf("  ground-state energy : %.10f\n", energy)
    @printf("  energy per site     : %.10f\n", energy / N)
    @printf("  total electrons     : %.6f  (half filling = %d)\n", sum(n_site), N)
    @printf("  mean double occ.    : %.6f\n", sum(docc) / N)
    @printf("  max bond dimension  : %d\n", maximum(bond_dimensions(psi)))
end

# ---------------------------------------------------------------------------
# 2D Hubbard lattice: same model on an Nx × Ny square lattice. Vertical bonds
# (chain range Nx) carry Jordan-Wigner strings, compiled by general_mpo.
# ---------------------------------------------------------------------------
let
    Nx, Ny, t, U = 2, 2, 1.0, 8.0
    N = Nx * Ny
    mu = U / 2
    H = hubbard_2d_mpo(Nx, Ny; t=t, U=U, mu=mu, T=Float64)
    psi0 = random_MPS(N, 4, 16; T=Float64, rng=rng)

    energy, psi = dmrg(H, psi0; nsweeps=12,
                       maxdim=[16, 32, 48],
                       cutoff=1e-10, tol=1e-8)

    ops = electron_operators(Float64)
    n_tot = sum(real(expect(psi, ops.Nup + ops.Ndn, i)) for i in 1:N)

    @printf("\n2D Hubbard lattice  %dx%d  t=%.1f  U=%.1f  (mu=U/2)\n", Nx, Ny, t, U)
    @printf("  ground-state energy : %.10f\n", energy)
    @printf("  energy per site     : %.10f\n", energy / N)
    @printf("  total electrons     : %.6f  (half filling = %d)\n", n_tot, N)
    @printf("  max bond dimension  : %d\n", maximum(bond_dimensions(psi)))
    @printf("  (MPO bond dimension grows ~O(Nx) from the vertical JW strings)\n")
end
