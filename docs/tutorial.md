# Tutorial: ground state of a Heisenberg chain

This tutorial runs the NaiveDMRG two-site solver, checks its result, and then
compares a small calculation with exact diagonalization and ITensor.

## 1. Activate the project

Start Julia in the repository root:

```bash
julia --project=.
```

Then load the package:

```julia
using NaiveDMRG
using LinearAlgebra
using Random
```

The from-scratch solver is the package's primary API, so `heisenberg_mpo`,
`random_MPS`, `dmrg`, `dense`, and friends are all exported directly. The
ITensor reference code is kept out of the way in the `NaiveDMRG.Reference`
submodule (see step 7).

## 2. Build the Hamiltonian

For an open spin-1/2 chain of length `N`, the implemented Hamiltonian is

```math
H = J \sum_{n=1}^{N-1}
    \left(S^x_n S^x_{n+1} + S^y_n S^y_{n+1} + S^z_n S^z_{n+1}\right)
    + h_z \sum_{n=1}^{N} S^z_n.
```

Construct its MPO with:

```julia
N = 20
H = heisenberg_mpo(N; J=1.0, hz=0.0)
```

The local physical dimension is two and the MPO bond dimension is five.

## 3. Build a reproducible initial MPS

```julia
rng = MersenneTwister(1234)
psi0 = random_MPS(N, 2, 16; rng=rng)

println(bond_dimensions(psi0))
println(norm(psi0))
```

`random_MPS` avoids impossible oversized bonds near the boundaries,
right-canonicalizes the state, and normalizes it. Supplying an RNG makes a run
reproducible.

## 4. Run two-site DMRG

```julia
energy, psi = dmrg(
    H,
    psi0;
    nsweeps=20,
    maxdim=64,
    cutoff=1e-10,
    tol=1e-8,
    eig_tol=1e-10,
    output=true,
)
```

A convergence line has this form:

```text
NaiveDMRG sweep 4: energy = -8.682...  delta = 2.1e-09  discarded = 3.4e-11
```

The quantities mean:

- `energy`: the normalized expectation value after both sweep directions;
- `delta`: absolute energy change from the preceding complete sweep;
- `discarded`: summed squared singular values removed during that sweep.

The solver stops early when `delta <= tol`. Reaching `nsweeps` is not itself a
guarantee of convergence, so inspect the final `delta` and repeat with larger
limits when necessary. `dmrg` optimizes a copy and leaves `psi0` untouched; use
`dmrg!` to optimize a state in place.

## 5. Check the result

```julia
@assert isapprox(norm(psi), 1.0; atol=1e-10)

measured_energy = compute_energy(H, psi)
@assert isapprox(measured_energy, energy; atol=1e-10)

println("E = ", energy)
println("bonds = ", bond_dimensions(psi))
```

To test bond-dimension convergence, repeat the calculation with increasing
`maxdim`, preferably starting each refinement from the previous optimized MPS:

```julia
for chi in (16, 32, 64, 128)
    global energy, psi = dmrg(
        H,
        psi;
        nsweeps=10,
        maxdim=chi,
        cutoff=1e-11,
        tol=1e-9,
        output=false,
    )
    println("maxdim=$chi  energy=$energy")
end
```

## 6. Validate a small system exactly

Never convert a production-sized state or operator to a dense object. For four
sites, however, dense conversion is a useful correctness check:

```julia
Nsmall = 4
Hsmall = heisenberg_mpo(Nsmall)
exact_energy = eigmin(Hermitian(dense(Hsmall)))

rng = MersenneTwister(8128)
psi_small = random_MPS(Nsmall, 2, 8; rng=rng)
dmrg_energy, psi_small = dmrg(
    Hsmall,
    psi_small;
    nsweeps=8,
    maxdim=8,
    cutoff=1e-12,
    tol=1e-10,
    output=false,
)

println("exact = ", exact_energy)
println("naive = ", dmrg_energy)
println("ΔE    = ", abs(dmrg_energy - exact_energy))
```

For this chain, the ground-state energy is approximately
`-1.6160254037844386`.

## 7. Compare with ITensor

ITensor is used here only to produce an independent reference value. Its DMRG
and Hamiltonian builders live in `NaiveDMRG.Reference`. Because ITensor exports
its own `MPS`/`MPO` types, import it qualified to avoid clashing with the
`NaiveDMRG.MPS`/`MPO` brought in by `using NaiveDMRG`:

```julia
import ITensors
import ITensorMPS

Nsmall = 4
sites = ITensorMPS.siteinds("S=1/2", Nsmall)
ops = NaiveDMRG.Reference.heisenberg_hamiltonian(Nsmall, 1, 1.0)
H_itensor = ITensorMPS.MPO(ops, sites)
initial = ITensorMPS.productMPS(sites, [isodd(i) ? "Up" : "Dn" for i in 1:Nsmall])

itensor_energy, _ = ITensorMPS.dmrg(
    H_itensor,
    initial;
    nsweeps=8,
    maxdim=8,
    cutoff=1e-12,
    outputlevel=0,
)

println("naive   = ", dmrg_energy)
println("ITensor = ", itensor_energy)
```

The regression test performs this three-way comparison automatically.

## 8. Construct an MPS manually

MPS tensors use `(left, physical, right)` ordering. A four-site product state
therefore has shapes `(1,2,1)` at every site:

```julia
up = reshape(ComplexF64[1, 0], 1, 2, 1)
down = reshape(ComplexF64[0, 1], 1, 2, 1)
neel = MPS([copy(up), copy(down), copy(up), copy(down)])

@assert norm(neel) ≈ 1
```

The constructors reject mismatched neighboring bonds and non-unit open
boundary bonds, catching common tensor-layout errors early.

## 9. Measure observables

A converged state is only useful once you can measure it. `expect` and
`correlation` evaluate normalized expectation values for any local `d×d`
operator; `spin_half_operators` provides the spin-1/2 matrices. The math is
derived in [theory §8](theory.md).

```julia
N = 12
H = heisenberg_mpo(N)
_, psi = dmrg(H, random_MPS(N, 2, 16); nsweeps=20, maxdim=48,
              cutoff=1e-11, tol=1e-9, output=false)

ops = spin_half_operators()

# Site-resolved magnetization ⟨Sᶻᵢ⟩ (real up to rounding for a Hermitian op).
sz = real.(expect(psi, ops.Sz))
println("⟨Sᶻ⟩ per site = ", round.(sz; digits=4))
println("total Sᶻ      = ", round(sum(sz); digits=8))   # ≈ 0 (singlet)

# Two-point spin correlation ⟨Sᶻ₁ Sᶻⱼ⟩.
c1j = [real(correlation(psi, ops.Sz, ops.Sz, 1, j)) for j in 1:N]
println("⟨Sᶻ₁ Sᶻⱼ⟩ = ", round.(c1j; digits=4))

# The full N×N correlation matrix, e.g. for a structure factor.
Czz = correlation_matrix(psi, ops.Sz, ops.Sz)
@assert isapprox(Czz, Czz'; atol=1e-8)          # ⟨SᶻᵢSᶻⱼ⟩ = ⟨SᶻⱼSᶻᵢ⟩
@assert all(isapprox(Czz[i,i], 0.25; atol=1e-8) for i in 1:N)  # ⟨(Sᶻᵢ)²⟩ = 1/4
```

Because operators are plain matrices, non-Hermitian correlators work too — for
example `correlation(psi, ops.Sp, ops.Sm, i, j)` returns the (generally complex)
$\langle S^+_i S^-_j\rangle$.

## Practical convergence checklist

Before trusting a result:

1. Confirm the energy change is below the requested tolerance.
2. Increase `maxdim` and verify the result is stable.
3. Reduce `cutoff` and `eig_tol` and verify stability.
4. Try more than one initial state when metastability is plausible.
5. Compare small versions of the model with exact diagonalization.
6. Compare representative cases with ITensor while the solver matures.
