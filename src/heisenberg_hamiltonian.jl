using ITensors

# Function to generate the Heisenberg Hamiltonian
function heisenberg_hamiltonian(Nx::Int, Ny::Int, J::Float64)
    N = Nx * Ny  # Total number of sites
    hamiltonian = OpSum()  # Initialize an empty operator sum
    
    for x in 1:Nx, y in 1:Ny
        site = (y-1)*Nx + x  # Convert 2D coordinates to 1D index
        
        # Horizontal interactions
        if x < Nx
            neighbor = site + 1  # Right neighbor
            # Add SxSx, SySy, and SzSz interactions
            hamiltonian += J, "Sx", site, "Sx", neighbor
            hamiltonian += J, "Sy", site, "Sy", neighbor
            hamiltonian += J, "Sz", site, "Sz", neighbor
        end
        
        # Vertical interactions
        if y < Ny
            neighbor = site + Nx  # Upper neighbor
            # Add SxSx, SySy, and SzSz interactions
            hamiltonian += J, "Sx", site, "Sx", neighbor
            hamiltonian += J, "Sy", site, "Sy", neighbor
            hamiltonian += J, "Sz", site, "Sz", neighbor
        end
    end
    
    return hamiltonian
end

