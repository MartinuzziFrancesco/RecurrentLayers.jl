using SafeTestsets
using Test

@safetestset "Quality Assurance" begin
    include("qa.jl")
end

@safetestset "Cells" begin
    include("test_cells.jl")
end

@safetestset "Layers" begin
    include("test_layers.jl")
end