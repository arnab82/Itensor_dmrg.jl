using ITensors, ITensorMPS

using LinearAlgebra

# The Hubbard Hamiltonian for a 2D square lattice
"""
    hubbard_hamiltonian(s::Vector{Index{Vector{Pair{QN,Int64}}}}, t::Float64, U::Float64)
    Args:
        s: Vector of site indices for the lattice with fermion conservation
        t: Hopping parameter
        U: On-site interaction parameter
    Returns:
        H: Hubbard Hamiltonian as an MPO
"""
function hubbard_hamiltonian(s::Vector{Index{Vector{Pair{QN,Int64}}}}, t::Float64, U::Float64)
    # Initialize an OpSum to store the terms of the Hamiltonian
    os = OpSum()
    
    # Loop over all sites to add on-site interaction terms
    for i in 1:N
        # Add U * n_up * n_down term for each site
        os += U, "Nupdn", i
    end
    
    # Loop over all sites to add hopping terms
    for i in 1:Nx
        for j in 1:Ny
            # Add horizontal hopping terms
            if i < Nx
                # Hopping term for spin-up electrons to the right
                os += -t, "Cdagup", i + (j-1)*Nx, "Cup", i+1 + (j-1)*Nx
                # Hopping term for spin-up electrons to the left
                os += -t, "Cdagup", i+1 + (j-1)*Nx, "Cup", i + (j-1)*Nx
                # Hopping term for spin-down electrons to the right
                os += -t, "Cdagdn", i + (j-1)*Nx, "Cdn", i+1 + (j-1)*Nx
                # Hopping term for spin-down electrons to the left
                os += -t, "Cdagdn", i+1 + (j-1)*Nx, "Cdn", i + (j-1)*Nx
            end
            
            # Add vertical hopping terms
            if j < Ny
                # Hopping term for spin-up electrons upwards
                os += -t, "Cdagup", i + (j-1)*Nx, "Cup", i + j*Nx
                # Hopping term for spin-up electrons downwards
                os += -t, "Cdagup", i + j*Nx, "Cup", i + (j-1)*Nx
                # Hopping term for spin-down electrons upwards
                os += -t, "Cdagdn", i + (j-1)*Nx, "Cdn", i + j*Nx
                # Hopping term for spin-down electrons downwards
                os += -t, "Cdagdn", i + j*Nx, "Cdn", i + (j-1)*Nx
            end
        end
    end
    
    # Convert the OpSum to an MPO (Matrix Product Operator)
    H = MPO(os, s)
    
    # Return the Hamiltonian as an MPO
    return H
end

