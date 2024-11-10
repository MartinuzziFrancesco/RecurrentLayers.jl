using SafeTestsets
using Test

@safetestset "Quality Assurance" begin
    include("qa.jl")
end

@safetestset "Sizes and parameters" begin
    include("test_cells.jl")
end