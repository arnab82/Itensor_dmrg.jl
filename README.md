# Itensor_dmrg.jl

A Julia package for performing Density Matrix Renormalization Group (DMRG) calculations on quantum many-body systems using ITensors.jl.

## Overview

This package provides implementations for:
- DMRG algorithms for finding ground states of quantum systems
- Heisenberg and Hubbard Hamiltonians
- Custom tensor operations and utilities
- Both ITensor-based and custom implementations

## Installation

### Prerequisites
- Julia 1.6 or higher
- Git (for cloning the repository)

### Install from GitHub

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arnab82/Itensor_dmrg.jl.git
   cd Itensor_dmrg.jl
   ```

2. **Activate and instantiate the project:**
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

   This will download and install all required dependencies including:
   - ITensors
   - ITensorMPS
   - KrylovKit
   - TensorOperations
   - And other dependencies listed in Project.toml

### Alternative: Install as a Package

You can also add this package directly in Julia:
```julia
using Pkg
Pkg.add(url="https://github.com/arnab82/Itensor_dmrg.jl.git")
```

## Running Tests

To verify the installation and run the test suite:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

Or from the command line:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Usage

### Basic Example: Heisenberg Model

```julia
using ITensors
using ITensorMPS
using Itensor_dmrg

# Define lattice parameters
Nx, Ny = 4, 4  # 4x4 lattice
N = Nx * Ny     # Total number of sites

# Define model parameter
J = 1.0  # Exchange coupling

# Create the Heisenberg Hamiltonian
H_heisenberg = Itensor_dmrg.heisenberg_hamiltonian(Nx, Ny, J)

# Define site indices for spin-1/2 system
s = siteinds("S=1/2", N)

# Convert OpSum to MPO
H = MPO(H_heisenberg, s)

# Create initial MPS (alternating up/down spins)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = randomMPS(s, state)

# Set up DMRG sweeps
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 100, 100, 200)
setcutoff!(sweeps, 1E-10)

# Run DMRG using ITensors built-in function
energy, ψ = dmrg(H, ψ, sweeps)
println("Ground state energy = ", energy)

# Or use the custom simple DMRG implementation
energy, ψ = Itensor_dmrg.simple_dmrg(H, ψ, 2; maxdim=10, cutoff=1E-6)
println("Ground state energy = ", energy)
```

### Hubbard Model Example

```julia
using ITensors
using ITensorMPS
using Itensor_dmrg

# Define lattice parameters
Nx, Ny = 4, 4
N = Nx * Ny

# Model parameters
t = 1.0  # Hopping parameter
U = 4.0  # On-site interaction

# Create site indices with quantum number conservation
sites = siteinds("Electron", N; conserve_qns=true)

# Create Hubbard Hamiltonian (MPO)
H = Itensor_dmrg.hubbard_hamiltonian(sites, t, U, Nx, Ny)

# Define initial state (half-filling with alternating spins)
state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
ψ = productMPS(sites, state)

# DMRG calculation
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 100, 100, 200)
setcutoff!(sweeps, 1E-10)

energy, ψ = dmrg(H, ψ, sweeps)
println("Ground state energy = ", energy)
```

### Custom DMRG Implementation

The package also includes a custom DMRG implementation with MPO and MPS classes:

```julia
include("src/custom/custom_dmrg.jl")

# Define system parameters
N = 20    # Number of sites
d = 4     # Physical dimension
chi = 15  # Bond dimension

# Create Hubbard Hamiltonian
Nx, Ny = 4, 4
t = 1.0
U = 4.0
H = hubbard(Nx=Nx, Ny=Ny, t=t, U=U, yperiodic=false)

# Initialize random MPS
mps = random_mps(Nx * Ny, 4, chi)

# Run DMRG
max_sweeps = 100
χ_max = 15
tol = 1e-6
energy, ground_state_mps = dmrg(H, mps, max_sweeps, χ_max, tol, hubbard)
println("Ground state energy: ", energy)
```

## Package Structure

```
Itensor_dmrg.jl/
├── src/
│   ├── Itensor/              # ITensor-based implementations
│   │   ├── Itensor_dmrg.jl  # Main module file
│   │   ├── heisenberg_hamiltonian.jl
│   │   ├── hubbard_hamiltonian.jl
│   │   ├── dmrg_ITensor.jl  # DMRG implementation
│   │   └── utils.jl         # Utility functions
│   └── custom/               # Custom tensor implementations
│       ├── custom_dmrg.jl   # Main custom module
│       ├── MPS.jl           # Matrix Product State
│       ├── MPO.jl           # Matrix Product Operator
│       ├── dmrg.jl          # Custom DMRG algorithm
│       ├── heisenberg_ham.jl
│       └── hubbard_ham.jl
├── example/                  # Example scripts
│   ├── heisenberg.jl
│   ├── hubbard.jl
│   └── test.jl
├── test/
│   └── runtests.jl          # Test suite
├── Project.toml             # Package dependencies
└── README.md                # This file
```

## Key Functions

### Hamiltonians
- `heisenberg_hamiltonian(Nx, Ny, J)`: Create Heisenberg model Hamiltonian
- `hubbard_hamiltonian(sites, t, U)`: Create Hubbard model Hamiltonian

### DMRG Functions
- `simple_dmrg(H, ψ, nsweeps; maxdim, cutoff)`: Perform DMRG optimization
- `compute_energy(ψ, H)`: Calculate energy expectation value

### Utilities
- `svd_truncate(ψ, b, phi; maxdim, cutoff, normalize, ortho)`: SVD with truncation

## Examples

See the `example/` directory for complete working examples:
- `heisenberg.jl`: Heisenberg model on a 4x4 lattice
- `hubbard.jl`: Hubbard model calculation
- `test.jl`: Custom implementation example

## Environment Caching

The package includes an optimized environment caching system for DMRG calculations. See `ENVIRONMENT_CACHING.md` for detailed documentation on:
- How environment caching works
- Performance benefits
- Handling aggressive bond dimension truncation

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Citation

If you use this package in your research, please cite:
```
@software{itensor_dmrg_jl,
  author = {Arnab},
  title = {Itensor_dmrg.jl: DMRG calculations with ITensors},
  url = {https://github.com/arnab82/Itensor_dmrg.jl},
  year = {2024}
}
```

## References

- ITensors.jl: https://github.com/ITensor/ITensors.jl
- DMRG Algorithm: White, S.R. (1992). "Density matrix formulation for quantum renormalization groups"
