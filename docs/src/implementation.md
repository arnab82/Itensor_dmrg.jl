# NaiveDMRG implementation

This document describes the current algorithm and the invariants relied on by
the from-scratch solver in `src/core/`. For the full step-by-step derivation of
the mathematics summarized here — the effective Hamiltonian as a projection, the
variational bound, and the truncation error — see the [theory notes](theory.md).

## Types and index conventions

An MPS site tensor is stored as

```text
A[left bond, physical state, right bond]
```

An MPO site tensor is stored as

```text
W[left MPO bond, physical output, physical input, right MPO bond]
```

Both structures use open boundaries. The first left bond and final right bond
must have dimension one. `MPS{T}` and `MPO{T}` are parametric in the scalar type
`T` (default `ComplexF64`); a real model built with `T=Float64` runs a fully
real solve.

## Canonicalization

`left_canonicalize!` reshapes each site to `(left * physical, right)`, applies a
QR decomposition, stores `Q` at the current site, and absorbs `R` into the next
site.

`right_canonicalize!` performs the mirrored operation by QR-factorizing the
adjoint of `(left, physical * right)`. DMRG begins from a normalized,
right-canonical state so the first left-to-right optimization has the expected
orthogonality center.

## Environments

The left environment after site `i` represents the contraction

```math
L_i = \langle \psi_{1:i} | H_{1:i} | \psi_{1:i} \rangle
```

with the MPS and MPO bonds at the cut left open. The right environment is the
corresponding contraction over sites to the right of a cut.

The environment arrays have index order

```text
(bra MPS bond, MPO bond, ket MPS bond).
```

`absorb_left` and `absorb_right` are the primitive contractions used for energy
evaluation and projected Hamiltonians.

## Two-site projected problem

At bond `i`, adjacent MPS tensors are contracted into

```text
theta[left bond, physical i, physical i+1, right bond].
```

The effective Hamiltonian is not formed as a dense matrix. Instead,
`effective_action` contracts a trial `theta` with:

- the left environment at the bond before site `i`;
- MPO tensors `W[i]` and `W[i+1]`;
- the right environment after site `i+1`.

KrylovKit calls this contraction as a linear map and solves for its lowest
algebraic eigenpair with Hermitian Lanczos iterations.

## Splitting and truncation

The optimized two-site tensor is reshaped to

```text
(left * physical i, physical i+1 * right)
```

and decomposed with an SVD. At most `maxdim` singular values are kept. When
`cutoff > 0`, additional values are discarded while their cumulative squared
weight remains no greater than

```math
\text{cutoff} \sum_j s_j^2.
```

During a right sweep, `U` is stored on the left site and `S V†` on the right;
during a left sweep, `U S` is stored on the left and `V†` on the right. This
moves the orthogonality center with the sweep direction.

## Sweep sequence

One complete sweep consists of:

1. optimize bonds `1` through `N-1` from left to right;
2. optimize bonds `N-1` through `1` from right to left;
3. contract the normalized variational energy;
4. compare it with the preceding complete-sweep energy.

The state supplied to `dmrg` is mutated. The returned state is normalized.

## Complexity

Let `d` be the physical dimension, `chi` the typical MPS bond dimension, and
`w` the MPO bond dimension. The dominant cost is repeated effective-Hamiltonian
application during local Krylov solves. Its exact contraction cost depends on
the contraction optimizer, but grows polynomially in `chi`, `d`, and `w`.

Memory is polynomial because the projected Hamiltonian is applied matrix-free.
In contrast, `dense(H)` requires a matrix of size `d^N by d^N`, and `dense(psi)`
requires a vector of length `d^N`.

## Validation

The canonical regression case is the four-site open antiferromagnetic
Heisenberg chain. The test checks:

- Hermiticity of the dense MPO;
- the known exact energy `-1.6160254037844386`;
- NaiveDMRG agreement with exact diagonalization;
- normalized final MPS and maximum bond dimension;
- NaiveDMRG agreement with the ITensor reference DMRG.

Run it through the package test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Optional U(1) block sparsity (`symmetry=true`)

Alongside the dense path above, `src/core/sym/` implements an optional
Abelian-U(1) block-sparse path, selected by `symmetry=true`. A `SymTensor` labels
every leg with charge sectors (an arrow `±1` and a `QN`), and stores only the
blocks whose signed charge sum equals a conserved flux; contraction, SVD, and QR
run block-by-block.

- `symmetrize_mpo` turns *any* charge-conserving dense MPO into a `SymMPO` by
  inferring each bond automaton state's `QN` via flux propagation, so the
  existing FSM builders are reused unchanged. `hubbard_mpo`, `hubbard_2d_mpo`,
  and `heisenberg_mpo` accept `symmetry=true`.
- `random_MPS(H::SymMPO; sector=...)` builds a `SymMPS` pinned to a target charge
  sector; `dmrg(H::SymMPO, psi::SymMPS)` dispatches to a block-sparse mirror of
  the sweep above (block environments, a packed sector-restricted local
  eigensolve, and block SVD truncation).

This is validated against the dense path (energies match exact diagonalization to
~1e-13). Its purpose is targeting a specific `(N↑, N↓)`/`Sz` sector — which the
dense solver cannot do — rather than speed: for `d = 4` the charge blocks are
small, so per-block overhead currently makes it slower than the tuned dense path.

## Current invariants and limitations

- Sites share one physical dimension `d`.
- Only open boundaries are accepted.
- The local effective operator is assumed Hermitian.
- Dense tensors are `Array{T}` with `T` a scalar-type parameter (real or
  complex); an optional U(1) block-sparse backend is available via
  `symmetry=true` (above).
- Public MPO builders: Heisenberg (`heisenberg_mpo`), generic nearest-neighbor
  (`nearest_neighbor_mpo`, `tfim_mpo`), the finite-range string compiler
  (`general_mpo`), and Fermi-Hubbard (`hubbard_mpo`, `hubbard_2d_mpo`).
- Dense conversion is a test utility only.
