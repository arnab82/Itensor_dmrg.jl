# NaiveDMRG.jl

`NaiveDMRG.jl` is an independent, finite-system, two-site DMRG implementation
for open-boundary matrix-product states, written from scratch in Julia. The
name reflects its intent: a small, readable, "naive" DMRG you can follow
end-to-end, rather than a production tensor-network framework.

ITensor is retained only as a reference implementation for regression tests and
numerical comparison; the solver itself never calls ITensor.

The implementation currently supports:

- dense complex MPS and MPO tensors;
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

- [Tutorial](docs/tutorial.md): a complete first calculation and validation
  workflow.
- [Theory](docs/theory.md): the algorithm derived step by step — MPS/MPO,
  canonical forms, environments, the effective Hamiltonian as a projection,
  SVD truncation, and the variational sweep. Shared-notation reference.
- [Implementation](docs/implementation.md): tensor definitions, environments,
  sweeps, truncation, and convergence, tied to the code.
- [Roadmap](docs/roadmap.md): current limitations and prioritized improvements.

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
