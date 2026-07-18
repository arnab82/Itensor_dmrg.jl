using NaiveDMRG
using Random

N = 25
rng = MersenneTwister(1234)
H = heisenberg_mpo(N)
psi0 = random_MPS(N, 2, 20; rng=rng)

energy, psi = dmrg(
    H,
    psi0;
    nsweeps=10,
    maxdim=40,
    cutoff=1e-10,
    tol=1e-8,
)

println("Final energy: ", energy)
println("Final bond dimensions: ", bond_dimensions(psi))
