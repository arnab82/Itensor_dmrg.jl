# Improvement roadmap

The matrix-free two-site algorithm is correct on the tested small-system
reference cases. The next work should broaden correctness coverage and stabilize
the API before adding sophisticated performance features.

## Already implemented

Several items originally listed below have since landed:

- **Structured results** — `dmrg`/`dmrg!` return a `DMRGResult` with per-sweep
  history, stopping reason, and local-solve residuals.
- **Sweep schedules** — `maxdim`, `cutoff`, `tol`, and `eig_tol` accept per-sweep
  tuples/vectors.
- **Explicit mutation** — `dmrg!` mutates in place; `dmrg` runs on a copy.
- **Incremental environments** — the sweep uses `right_sweep_cache!` plus in-place
  `absorb_left!`/`absorb_right!`; it no longer rebuilds all environments each
  half-sweep.
- **Generic nearest-neighbor MPO builder** — `nearest_neighbor_mpo`, `tfim_mpo`.
- **Local observables and correlations** — `expect`, `correlation`,
  `correlation_matrix`, `spin_half_operators`.
- **Entanglement diagnostics** — `schmidt_values`, `entanglement_entropy`.
- **Single-site DMRG with subspace expansion** — `single_site_dmrg`,
  `single_site_dmrg!`.

Completed subsections below are marked **(Done.)**. The math for the from-scratch
solver, its observables, entanglement, and the MPO builder is derived in
[theory.md](theory.md).

## Priority 0: correctness and repository consistency

### Expand contraction-level tests

Test every mathematical primitive against dense linear algebra on two to six
sites:

- left and right environment contractions;
- effective-Hamiltonian action;
- MPS overlap and normalization;
- MPO-to-dense conversion;
- left/right canonical conditions;
- discarded-weight selection;
- nonzero field `hz` and complex Hermitian MPOs.

These tests localize tensor-index mistakes much better than an end-to-end energy
test alone.

### Test more chains, seeds, and bond limits

Add exact-energy tests for multiple lengths and random seeds. Include tests where
`maxdim` intentionally restricts the state, verifying the variational upper
bound and improvement as `maxdim` increases.

### Handle local eigensolver failure explicitly

The current code warns when KrylovKit does not converge but still uses the
returned vector. A production API should either retry with a larger Krylov
space, fall back to a dense solve for small local problems, or return a failed
status to the caller.

### Remove or migrate legacy code

The old single-site and custom-Hubbard example/test executables have been
removed, and the ITensor code now lives in the isolated `NaiveDMRG.Reference`
submodule. What remains is to port genuinely useful reference features (for
example a validated Hubbard MPO) onto the new `(left, physical, right)` tensor
conventions instead of leaving them ITensor-only.

## Priority 1: stable user API

### Return structured results — **(Done.)**

`DMRGResult` supplements the `(energy, psi)` destructuring with a result object
containing:

- final energy and state;
- convergence flag and stopping reason;
- energy and discarded weight per sweep;
- maximum bond dimension per sweep;
- local eigensolver iteration/residual data;
- elapsed time and allocation statistics.

This makes convergence auditable and enables plotting without parsing output.

### Add sweep schedules — **(Done.)**

`maxdim`, `cutoff`, `tol`, and `eig_tol` accept scalars or per-sweep schedules
(tuples/vectors). Optional per-sweep noise is still open. Gradually increasing
the bond dimension is often more efficient and robust than starting at the final
maximum.

### Make mutation explicit — **(Done.)**

`dmrg!` is the mutating implementation and `dmrg` is the copying wrapper.

### Add model-independent construction tools — **(Partly done.)**

Supported builders now cover:

- generic nearest-neighbor MPOs — `nearest_neighbor_mpo`, `tfim_mpo` **(Done)**;
- local expectation values and correlation functions — `expect`, `correlation`
  **(Done)**.

Still open:

- a product-MPS / product-state builder;
- longer-range and finite-state-machine MPOs beyond nearest neighbor;
- the Hubbard model after fermionic sign conventions are verified.

### Make ITensor an optional dependency

The from-scratch core is already separated from the ITensor comparison code:
the solver is the top-level `NaiveDMRG` API and ITensor lives in the
`NaiveDMRG.Reference` submodule. However, the root package still loads ITensor
unconditionally through that submodule. Moving `Reference` behind a package
extension (loaded only when a user has ITensor available) would cut load time
and make the independence of the solver complete rather than merely namespaced.

## Priority 2: performance and scalability

### Avoid rebuilding unused environments — **(Done.)**

`dmrg!` seeds each sweep with `right_sweep_cache!` and then updates only the one
environment each local step consumes, via in-place `absorb_left!`/`absorb_right!`.
It no longer constructs every environment before each half-sweep.

### Reuse contraction workspaces

`effective_action`, environment updates, SVD splitting, and energy evaluation
allocate new arrays repeatedly. Introduce reusable buffers keyed by tensor
shape, use in-place multiplication where practical, and benchmark allocations
per local solve.

### Parameterize scalar and storage types

Change `MPS`, `MPO`, and environments from hard-coded `ComplexF64` arrays to
parametric scalar/storage types. This allows real Hamiltonians to use
`Float64`, supports alternative precision, and creates a path toward GPU or
block-sparse arrays.

### Tune contraction order and eigensolver settings

Benchmark effective-Hamiltonian application over representative `(d, chi, w)`
shapes. Precompute an optimized contraction plan when shapes repeat, expose
Krylov dimension and iteration limits, and record residuals and matvec counts.

### Add reproducible benchmarks

Replace ad hoc timing scripts with benchmark cases that report:

- time and allocations per effective-Hamiltonian application;
- time per half-sweep;
- peak environment memory;
- scaling with chain length and bond dimension;
- comparison with ITensor under matched numerical settings.

Performance comparisons should verify equal energies and tolerances first.

## Priority 3: advanced DMRG features

After the core API and tests stabilize:

- quantum-number-conserving block-sparse tensors;
- single-site DMRG with subspace expansion **(Done — `single_site_dmrg`)**;
- excited states through orthogonality penalties or projected solvers;
- finite-state/MPO builders for long-range interactions;
- periodic-boundary support, with explicit cost expectations;
- checkpoint/restart and observer callbacks;
- CPU threading and GPU backends;
- automatic differentiation where contraction backends support it.

## Recommended next milestone

With structured results, sweep schedules, incremental environments, the generic
MPO builder, observables, entanglement diagnostics, and single-site DMRG all in
place, the next focused steps are:

1. **Parametric scalar/storage types** — generalize `MPS`, `MPO`, and the
   environments off hard-coded `ComplexF64` so real Hamiltonians run in `Float64`
   (Priority 2). This is the widest-reaching remaining refactor.
2. **Remaining construction tools** — a product-state builder and a validated,
   sign-correct Hubbard MPO on the native tensor conventions.
3. **Excited states** — via orthogonality penalties or projected solvers
   (Priority 3).

Deferred for now: explicit local-eigensolver failure handling, and moving
`NaiveDMRG.Reference` behind a package extension.
