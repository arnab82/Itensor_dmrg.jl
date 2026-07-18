# Improvement roadmap

The matrix-free two-site algorithm is now correct on the tested small-system
reference case. The next work should broaden correctness coverage and stabilize
the API before adding sophisticated performance features.

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

### Return structured results

Replace the bare `(energy, psi)` result, or supplement it, with a result object
containing:

- final energy and state;
- convergence flag and stopping reason;
- energy and discarded weight per sweep;
- maximum bond dimension per sweep;
- local eigensolver iteration/residual data;
- elapsed time and allocation statistics.

This makes convergence auditable and enables plotting without parsing output.

### Add sweep schedules

Accept scalar values or per-sweep schedules for `maxdim`, `cutoff`, `eig_tol`,
and optional noise. Gradually increasing the bond dimension is often more
efficient and robust than starting at the final maximum.

### Make mutation explicit

Keep `dmrg!` for the mutating implementation and provide `dmrg` as a copying
wrapper. At present `dmrg` mutates its input, which can surprise users and makes
comparative experiments easier to contaminate accidentally.

### Add model-independent construction tools

Provide supported builders for:

- product MPSs;
- sums of local operator terms;
- generic nearest-neighbor MPOs;
- the Hubbard model after fermionic sign conventions are verified;
- local expectation values and correlation functions.

The DMRG engine is generic over compatible `MPO`s, but users currently lack a
safe public route to construct most of them.

### Make ITensor an optional dependency

The from-scratch core is already separated from the ITensor comparison code:
the solver is the top-level `NaiveDMRG` API and ITensor lives in the
`NaiveDMRG.Reference` submodule. However, the root package still loads ITensor
unconditionally through that submodule. Moving `Reference` behind a package
extension (loaded only when a user has ITensor available) would cut load time
and make the independence of the solver complete rather than merely namespaced.

## Priority 2: performance and scalability

### Avoid rebuilding unused environments

`sweep!` currently calls `environments`, which constructs every left and right
environment before each half-sweep. A left-to-right sweep needs the initial
right environments and incrementally updated left environments; the reverse
sweep needs the opposite. Building only what each direction consumes removes
roughly one redundant environment pass per half-sweep.

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
- single-site DMRG with subspace expansion;
- excited states through orthogonality penalties or projected solvers;
- finite-state/MPO builders for long-range interactions;
- periodic-boundary support, with explicit cost expectations;
- checkpoint/restart and observer callbacks;
- CPU threading and GPU backends;
- automatic differentiation where contraction backends support it.

## Recommended next milestone

A focused next release should complete the Priority 0 items, add a structured
result/history type, implement `dmrg!` plus a copying `dmrg`, and eliminate
redundant environment construction. That combination improves trustworthiness,
usability, and speed without changing the underlying algorithm.
