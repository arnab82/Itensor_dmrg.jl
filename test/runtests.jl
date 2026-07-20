using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

# Import (not `using`) so the module name is available for qualified access
# without pulling NaiveDMRG.MPS/MPO into scope, where they would collide with
# the ITensor `MPS`/`MPO` types the reference tests rely on.
import NaiveDMRG

@testset "NaiveDMRG.jl" begin
    include("naive_primitives_test.jl")
    include("naive_dmrg_test.jl")
    include("single_site_dmrg_test.jl")
    include("mpo_builder_test.jl")
    include("observables_test.jl")
    include("entanglement_test.jl")
    include("parametric_types_test.jl")
    include("reference_test.jl")
end
