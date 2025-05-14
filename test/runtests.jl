using SafeTestsets, Test

@safetestset "Quality Assurance" begin
    include("qa.jl")
end

@safetestset "Cells" begin
    include("test_cells.jl")
end

@safetestset "Layers" begin
    include("test_layers.jl")
end

@safetestset "Wrappers" begin
    include("test_layer_wrappers.jl")
    include("test_cell_wrappers.jl")
end
