# Pure-ITensor Heisenberg example. It uses NaiveDMRG only for the reference
# OpSum builder in NaiveDMRG.Reference; the from-scratch solver is demonstrated
# in 25_site_dmrg_demo.jl. `import` avoids clashing NaiveDMRG.MPS/MPO with
# ITensor's own MPS/MPO types.
using ITensors, ITensorMPS
import NaiveDMRG

let
    # Define lattice parameters
    Nx, Ny = 3, 3
    N = Nx * Ny

    # Define model parameter
    J = 1.234

    # Create the Heisenberg Hamiltonian
    H_heisenberg = NaiveDMRG.Reference.heisenberg_hamiltonian(Nx, Ny, J)

    # Define site indices for the lattice
    s = ITensors.siteinds("S=1/2", N)

    # Convert OpSum to ITensorMPS.MPO explicitly; this stays correct even if
    # Main already has NaiveDMRG.MPO bound from an earlier REPL command.
    H = ITensorMPS.MPO(H_heisenberg, s)

    # Create a product initial MPS in the Sz = 0 sector
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    ψ = ITensorMPS.productMPS(s, state)

    # DMRG parameters
    sweeps = ITensorMPS.Sweeps(10)
    ITensorMPS.setmaxdim!(sweeps, 10, 20, 50, 100, 100)
    ITensorMPS.setcutoff!(sweeps, 1E-10)

    # Run DMRG
    energy, ψ = ITensorMPS.dmrg(H, ψ, sweeps)

    println("Ground state energy = ", energy)

    # Calculate magnetization
    Sz_total = sum(ITensorMPS.expect(ψ, "Sz"))
    println("Total Sz = ", Sz_total)

    # Optionally run the ITensor reference simple_dmrg for comparison
    # ψ2, energy2 = NaiveDMRG.Reference.simple_dmrg(H, ψ, 2; maxdim=50, cutoff=1E-8)
    # println("Ground state energy (simple_dmrg) = ", energy2)

    display(ψ)
end
