# NaiveDMRG.jl

`NaiveDMRG.jl` is an independent, finite-system DMRG implementation for
open-boundary matrix-product states, written from scratch in Julia. The name
reflects its intent: a small, readable, "naive" DMRG you can follow end-to-end,
rather than a production tensor-network framework.

ITensor is retained only as a reference implementation for regression tests and
numerical comparison (in the `NaiveDMRG.Reference` submodule); the solver itself
never calls ITensor.

## Features

- dense complex MPS and MPO tensors with a fixed index order;
- left and right canonicalization;
- matrix-free two-site effective Hamiltonians and KrylovKit local eigensolves;
- single-site DMRG with subspace expansion;
- SVD bond truncation by maximum dimension and discarded weight;
- an open spin-1/2 Heisenberg MPO, a generic nearest-neighbor MPO builder
  (`nearest_neighbor_mpo`, `tfim_mpo`), and a finite-range MPO compiler with
  operator strings (`general_mpo`);
- validated 1D and 2D Fermi-Hubbard models with Jordan-Wigner fermion signs
  (`hubbard_mpo`, `hubbard_2d_mpo`, `electron_operators`);
- an optional Abelian-U(1) block-sparse path (`symmetry=true`) that targets a
  chosen `(N↑, N↓)` or `Sz` charge sector directly;
- local observables and correlation functions (`expect`, `correlation`,
  `correlation_matrix`);
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

## Quick start

The from-scratch solver is the package's primary API and is exported directly:

```julia
using NaiveDMRG
using Random

H = heisenberg_mpo(20; J = 1.0)
psi0 = random_MPS(20, 2, 16; rng = MersenneTwister(1234))

energy, psi = dmrg(H, psi0; nsweeps = 20, maxdim = 64, cutoff = 1e-10, tol = 1e-8)

println("energy = ", energy)
println("bond dimensions = ", bond_dimensions(psi))
```

## Where to go next

- The [Tutorial](tutorial.md) walks through a complete first calculation,
  measuring observables and entanglement, and validating against exact
  diagonalization and ITensor.
- The [Theory](theory.md) notes derive the algorithm step by step — MPS/MPO,
  canonical forms (and *why* we canonicalize and normalize), environments, the
  effective Hamiltonian as a projection, SVD truncation, the variational sweep,
  subspace expansion, observables, and entanglement.
- The [Implementation](implementation.md) notes tie the algorithm to the code.
- The [API reference](api.md) lists the exported functions and types.
