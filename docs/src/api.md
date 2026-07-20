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
