# NaiveDMRG.jl

[![CI](https://github.com/arnab82/NaiveDMRG.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/arnab82/NaiveDMRG.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://arnab82.github.io/NaiveDMRG.jl/)

`NaiveDMRG.jl` is an independent, finite-system DMRG implementation for
open-boundary matrix-product states, written from scratch in Julia. The name
reflects its intent: a small, readable, "naive" DMRG you can follow end-to-end,
rather than a production tensor-network framework.

📖 **Live documentation: <https://arnab82.github.io/NaiveDMRG.jl/>**

ITensor is retained only as a reference implementation for regression tests and
numerical comparison; the solver itself never calls ITensor.

The implementation currently supports:

- dense MPS and MPO tensors with a parametric scalar type — `MPS{T}`/`MPO{T}`,
  `ComplexF64` by default or real `Float64` for a fully real solve;
- left and right canonicalization;
- matrix-free two-site effective Hamiltonians;
- single-site DMRG with subspace expansion (`single_site_dmrg`);
- KrylovKit local eigensolves;
- SVD bond truncation by maximum dimension and discarded weight;
- an open spin-1/2 Heisenberg-chain MPO, plus a generic nearest-neighbor MPO
  builder (`nearest_neighbor_mpo`, `tfim_mpo`) for arbitrary 1D models;
- local observables and two-point correlation functions (`expect`,
  `correlation`, `correlation_matrix`);
- bipartite entanglement entropy and the Schmidt spectrum
  (`entanglement_entropy`, `schmidt_values`);
- an optional Abelian-U(1) block-sparse path (`symmetry=true`) that targets a
  chosen `(N↑, N↓)` or `Sz` charge sector (see below);
- comparison with exact diagonalization and ITensor on small systems.

## Installation

From the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Run the complete test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Quick start

The from-scratch solver is the package's primary API and is exported directly:

```julia
using NaiveDMRG
using Random

rng = MersenneTwister(1234)

H = heisenberg_mpo(20; J=1.0)
psi0 = random_MPS(20, 2, 16; rng=rng)

energy, psi = dmrg(
    H,
    psi0;
    nsweeps=20,
    maxdim=64,
    cutoff=1e-10,
    tol=1e-8,
)

println("energy = ", energy)
println("bond dimensions = ", bond_dimensions(psi))
```

`dmrg` runs on a copy and returns `(energy, psi)`; use `dmrg!` to optimize a
state in place.

For a real model, pass `T=Float64` to build a real MPO and state — the whole
solve then runs without complex arithmetic:

```julia
H = heisenberg_mpo(20; T=Float64)
psi0 = random_MPS(20, 2, 16; T=Float64)
energy, psi = dmrg(H, psi0; nsweeps=20, maxdim=64)   # eltype(psi) == Float64
```

## U(1) symmetry: targeting a charge sector

By default the solver carries no quantum-number symmetry, so `dmrg` finds the
**global** ground state over all fillings (for Hubbard this is why half filling
is reached via the particle-hole point `mu = U/2`). Passing `symmetry=true` to a
model builder instead produces an Abelian-U(1) **block-sparse** operator, and a
matching `random_MPS` is pinned to a chosen charge sector — so DMRG can target a
**specific** `(N↑, N↓)` (Hubbard) or `Sz` (spin-1/2) sector directly:

```julia
N = 8
H = hubbard_mpo(N; U=8.0, mu=0.0, T=Float64, symmetry=true)   # SymMPO

# ground state at exactly 3 up + 3 down electrons (two holes off half filling)
psi0 = random_MPS(H, 16; sector=QN(3, 3), T=Float64)
energy, psi = dmrg(H, psi0; nsweeps=20, maxdim=64)

ops = electron_operators(Float64)
sum(real(expect(psi, ops.Nup, i)) for i in 1:N)   # == 3.0, exactly
```

Conserved charges are `(N↑, N↓)` for the electron site and `2·Sz` for spin-1/2;
`electron_half_filling(N)` is a convenience for the `(N/2, N/2)` sector. See
`example/hubbard_symmetry.jl`.

The symmetric path is **correct** (energies match exact diagonalization and the
dense solver to ~1e-13) and its unique capability is fixing the charge sector.
It is **not** currently a speedup: for these small `d = 4` chains the charge
blocks are tiny, so per-block overhead makes it somewhat slower and only modestly
lighter on memory than the tuned dense path. Block-sparsity pays off on wall time
only at large per-sector bond dimension (wide 2D), which this from-scratch code
is not tuned for.

## Tensor conventions

The solver uses a fixed index order:

- MPS tensor: `(left_bond, physical, right_bond)`
- MPO tensor: `(left_bond, physical_out, physical_in, right_bond)`

The `MPS` and `MPO` constructors validate open boundaries, physical dimensions,
and neighboring bond dimensions.

## Choosing parameters

- `nsweeps`: maximum number of complete right-and-left sweeps.
- `maxdim`: largest retained MPS bond dimension.
- `cutoff`: maximum relative discarded singular-value weight at each split.
- `tol`: convergence threshold for the change in the normalized energy between
  complete sweeps.
- `eig_tol`: tolerance for each local Krylov eigensolve.
- `output`: print one convergence line per complete sweep.

Each of `maxdim`, `cutoff`, `tol`, and `eig_tol` also accepts a per-sweep
schedule (tuple or vector); the last entry is reused for any further sweeps.
Start with a modest `maxdim`, check convergence, and increase it until the
energy and observables of interest stop changing at the required precision.

## Documentation

The full documentation is published at
<https://arnab82.github.io/NaiveDMRG.jl/>. The sources live in `docs/src/`:

- [Tutorial](docs/src/tutorial.md): a complete first calculation and validation
  workflow.
- [Theory](docs/src/theory.md): the algorithm derived step by step — MPS/MPO,
  canonical forms, environments, the effective Hamiltonian as a projection,
  SVD truncation, the variational sweep, subspace expansion, observables, and
  entanglement. Shared-notation reference.
- [Implementation](docs/src/implementation.md): tensor definitions, environments,
  sweeps, truncation, and convergence, tied to the code.
- [Roadmap](docs/src/roadmap.md): current status and prioritized improvements.

## Current scope

The supported model constructor is the open spin-1/2 Heisenberg chain. The DMRG
engine itself accepts any compatible `MPO`, but a stable public builder for
general Hamiltonians is not implemented yet.

The ITensor reference code lives in the `NaiveDMRG.Reference` submodule
(`NaiveDMRG.Reference.heisenberg_hamiltonian`, `.simple_dmrg`, and so on). It is
used only by the comparison tests and examples and is never invoked by the
exported `NaiveDMRG` API.

Full dense conversion through `dense` grows exponentially and is intended only
for small-system validation.
