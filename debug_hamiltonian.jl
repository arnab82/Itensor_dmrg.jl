#!/usr/bin/env julia
"""
Debug: Print out the Hamiltonian matrix for 3-site system
"""

using LinearAlgebra

include("./src/custom/custom_dmrg.jl")

N = 3
d = 2
χ_mpo = 5

println("Creating 3-site Heisenberg Hamiltonian...")
H = heisenberg_ham(N, d, χ_mpo)

println("\nConverting to matrix...")
H_matrix = MPO_to_array(H)

println("\nHamiltonian matrix ($(size(H_matrix))):")
display(H_matrix)

println("\n\nHermiticity check:")
hermitian_error = norm(H_matrix - H_matrix') / norm(H_matrix)
println("  Relative error: $hermitian_error")

println("\nEigenvalues:")
eigenvalues = sort(real(eigvals(Hermitian((H_matrix + H_matrix') / 2))))
for (i, e) in enumerate(eigenvalues)
    println("  $i: $e")
end

println("\nGround state energy: $(eigenvalues[1])")
