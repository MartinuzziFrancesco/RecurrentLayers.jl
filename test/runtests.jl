using SafeTestsets
using Test

@safetestset "Quality Assurance" begin
    include("qa.jl")
end

@safetestset "Minimal gated unit" begin
    include("mgu_cell.jl")
end

@safetestset "Independently recurrent neural network" begin
    include("indrnn_cell.jl")
end

@safetestset "Light gated recurrent unit" begin
    include("ligru_cell.jl")
end

@safetestset "Light recurrent unit" begin
    include("lightru_cell.jl")
end

@safetestset "Recurrent additive network" begin
    include("ran_cell.jl")
end