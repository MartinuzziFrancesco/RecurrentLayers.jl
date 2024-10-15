using SafeTestsets
using Test

@safetestset "Quality Assurance" begin
    include("qa.jl")
end
