using ITensors
using Printf
using KrylovKit

function compute_energy(psi::MPS, H::MPO)
    # Permute the indices of H for better memory layout
    H = permute(H, (linkind, siteinds, linkind))
    # Calculate the energy expectation value
    energy = real(inner(psi', H, psi))

    return energy
end
#ProjMPO , position! functions are taken from ITensor library
"""
A ProjMPO computes and stores the projection of an
MPO into a basis defined by an MPS, leaving a
certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling
the `position!` method.

Drawing of the network represented by a ProjMPO `P(H)`,
showing the case of `nsite(P)==2` and `position!(P,psi,4)`
for an MPS `psi`:

```
o--o--o-      -o--o--o--o--o--o <psi|
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o--o--o--o--o--o--o--o H
|  |  |  |  |  |  |  |  |  |  |
o--o--o-      -o--o--o--o--o--o |psi>
```
"""

"""
    position!(P::ProjMPOSum, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPOSum `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPOs on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""

"""
    simple_dmrg(H::MPO, psi0::MPS, nsweeps::Int; maxdim::Int=50, cutoff::Float64=1E-8)
    args:
        H: MPO
        psi0: MPS
        nsweeps: Int
        maxdim: Int
        cutoff: Float64

    Perform simple DMRG algorithm to find the ground state of a 1D quantum system.
    returns:
        psi: MPS
        energy: Float64
"""


function simple_dmrg(H::MPO, psi0::MPS, nsweeps::Int; maxdim::Int=50, cutoff::Float64=1E-8,
                    eigsolve_tol::Float64=1e-14, eigsolve_krylovdim::Int=3,
                    eigsolve_maxiter::Int=1, eigsolve_verbosity::Int=0,
                    ishermitian::Bool=true, silent::Bool=false)
    ITensors.set_warn_order(100)
    # Create a copy of the initial MPS to avoid modifying the input
    psi = copy(psi0)
    # Get the number of sites in the MPS
    N = length(psi)
    #orthogonalize the MPS
    if !isortho(psi) || orthocenter(psi) != 1
        psi = ITensors.orthogonalize(psi, 1)
    end
    
    # Create ProjMPO once and reuse it (major memory optimization)
    PH = ProjMPO(H)
    
    # Calculate initial memory usage (only if not silent)
    if !silent
        initial_mem = sum(sizeof, psi) + sizeof(H)
        println("Initial memory usage: $(initial_mem / 1e6) MB")
    end

    # Main DMRG sweep loop
    for sweep in 1:nsweeps
        # Sweep from left to right
        for b in 1:(N-1)
            !silent && println("Sweep $sweep, bond $b, left-to-right")
            # Combine tensors at sites b and b+1
            wf = psi[b] * psi[b+1]
            # Position the projector at site b (reuse PH object)
            position!(PH, psi, b)
            # Solve for the ground state of the effective Hamiltonian
            vals, vecs = eigsolve(PH, wf, 1, :SR; 
                                   ishermitian=ishermitian,
                                   tol=eigsolve_tol,
                                   krylovdim=eigsolve_krylovdim,
                                   maxiter=eigsolve_maxiter,
                                   verbosity=eigsolve_verbosity)
            # Extract the ground state energy and eigenvector
            energy = vals[1]
            eigenvector = vecs[1]
            # Perform SVD and truncation, updating the MPS
            psi = svd_truncate(psi, b, eigenvector; maxdim=maxdim, cutoff=cutoff, normalize=true, ortho="left")
        end

        # Sweep from right to left
        for b in (N-1):-1:1
            !silent && println("Sweep $sweep, bond $b, right-to-left")
            wf = psi[b] * psi[b+1]
            # Reuse PH object, just reposition it
            position!(PH, psi, b)
            vals, vecs = eigsolve(PH, wf, 1, :SR; 
                                   ishermitian=ishermitian,
                                   tol=eigsolve_tol,
                                   krylovdim=eigsolve_krylovdim,
                                   maxiter=eigsolve_maxiter,
                                   verbosity=eigsolve_verbosity)
            energy = vals[1]
            eigenvector = vecs[1]
            psi = svd_truncate(psi, b, eigenvector; maxdim=maxdim, cutoff=cutoff, normalize=true, ortho="right")
        end

        # Calculate current memory usage (only if not silent)
        if !silent
            current_mem = sum(sizeof, psi) + sizeof(H)
            println("Memory usage after sweep $sweep: $(current_mem / 1e6) MB")
        end

        # Print the energy after each sweep
        @printf("Sweep %d Energy: %.12f\n", sweep, real(energy))
    end

    # Return the final MPS and the real part of the energy
    return psi, real(energy)
end