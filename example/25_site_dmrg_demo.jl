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
    maxdim=[10,10,10, 20,20, 20,20,40],
    cutoff=1e-10,
    tol=1e-8,
)

println("Final energy: ", energy)
println("Final bond dimensions: ", bond_dimensions(psi))

N = 100
rng = MersenneTwister(1234)
H = heisenberg_mpo(N)
psi0 = random_MPS(N, 2, 20; rng=rng)

energy, psi = dmrg(
    H,
    psi0;
    nsweeps=12,
    maxdim=[10,20,40, 80,160, 320,320,400,400],
    cutoff=1e-10,
    tol=1e-8,
)

println("Final energy: ", energy)
println("Final bond dimensions: ", bond_dimensions(psi))
