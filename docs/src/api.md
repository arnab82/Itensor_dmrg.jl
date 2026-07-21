# API reference

The from-scratch solver's public API, exported from `NaiveDMRG`. The ITensor
reference code lives in the [`NaiveDMRG.Reference`](@ref) submodule and is not
part of this interface.

```@meta
CurrentModule = NaiveDMRG
```

## Module

```@docs
NaiveDMRG
```

## Types and states

```@docs
MPS
MPO
random_MPS
dense
```

## Hamiltonians

```@docs
heisenberg_mpo
nearest_neighbor_mpo
tfim_mpo
general_mpo
```

## Fermi-Hubbard models

```@docs
hubbard_mpo
hubbard_2d_mpo
electron_operators
```

## U(1) symmetry (charge sectors)

The optional block-sparse path. Passing `symmetry=true` to a model builder returns
a [`SymMPO`](@ref); pair it with a [`random_MPS`](@ref) seeded in a target sector
(or [`random_sym_mps`](@ref)) and call [`dmrg`](@ref), which dispatches to the
symmetric engine. See the tutorial's charge-sector section.

```@docs
QN
SymMPS
SymMPO
symmetrize_mpo
random_sym_mps
electron_site_qns
spinhalf_site_qns
electron_half_filling
```

## Ground-state solvers

```@docs
dmrg
dmrg!
single_site_dmrg
single_site_dmrg!
DMRGResult
SweepRecord
LocalSolveRecord
```

## Observables and entanglement

```@docs
compute_energy
expect
correlation
correlation_matrix
spin_half_operators
schmidt_values
entanglement_entropy
```

## Reference submodule

```@docs
NaiveDMRG.Reference
```
