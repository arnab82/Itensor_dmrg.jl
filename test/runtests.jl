using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Itensor_dmrg

@testset "Itensor_dmrg.jl Tests" begin
    
    @testset "Heisenberg Hamiltonian Tests" begin
        # Test Heisenberg Hamiltonian construction
        @testset "Small lattice Heisenberg" begin
            Nx, Ny = 2, 2
            N = Nx * Ny
            J = 1.0
            
            # Create the Hamiltonian
            H_heisenberg = Itensor_dmrg.heisenberg_hamiltonian(Nx, Ny, J)
            
            # Check that we get an OpSum
            @test H_heisenberg isa OpSum
            
            # Create site indices and MPO
            s = siteinds("S=1/2", N)
            H = MPO(H_heisenberg, s)
            
            # Check MPO properties
            @test length(H) == N
            @test H isa MPO
        end
    end
    
    @testset "Hubbard Hamiltonian Tests" begin
        @testset "Small lattice Hubbard" begin
            Nx, Ny = 2, 2
            N = Nx * Ny
            t = 1.0
            U = 4.0
            
            # Create site indices with quantum number conservation
            sites = siteinds("Electron", N; conserve_qns=true)
            
            # Create the Hamiltonian
            H = Itensor_dmrg.hubbard_hamiltonian(sites, t, U, Nx, Ny)
            
            # Check MPO properties
            @test H isa MPO
            @test length(H) == N
        end
    end
    
    @testset "DMRG Function Availability" begin
        @testset "simple_dmrg function exists" begin
            # Test that the simple_dmrg function is defined and exported
            @test isdefined(Itensor_dmrg, :simple_dmrg)
            @test hasmethod(Itensor_dmrg.simple_dmrg, (MPO, MPS, Int))
        end
    end
    
    @testset "Utility Functions Tests" begin
        @testset "Energy computation" begin
            # Create a simple test system
            N = 4
            s = siteinds("S=1/2", N)
            
            J = 1.0
            H_opsum = Itensor_dmrg.heisenberg_hamiltonian(2, 2, J)
            H = MPO(H_opsum, s)
            
            # Create a product state
            state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
            ψ = productMPS(s, state)
            
            # Compute energy
            E = Itensor_dmrg.compute_energy(ψ, H)
            
            # Check that energy is a real number
            @test E isa Real
            @test isfinite(E)
        end
        
        @testset "SVD truncate" begin
            # Create a simple MPS to test SVD truncation
            N = 4
            s = siteinds("S=1/2", N)
            ψ = randomMPS(s; linkdims=10)
            
            # Get two-site wavefunction
            b = 2
            phi = ψ[b] * ψ[b+1]
            
            # Test left orthogonalization
            ψ_new = Itensor_dmrg.svd_truncate(ψ, b, phi; maxdim=5, cutoff=1E-10, normalize=true, ortho="left")
            @test ψ_new isa MPS
            @test length(ψ_new) == N
            
            # Test right orthogonalization
            ψ_new2 = Itensor_dmrg.svd_truncate(ψ, b, phi; maxdim=5, cutoff=1E-10, normalize=true, ortho="right")
            @test ψ_new2 isa MPS
            @test length(ψ_new2) == N
        end
    end
end
