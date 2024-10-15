using RecurrentLayers
using Documenter

DocMeta.setdocmeta!(RecurrentLayers, :DocTestSetup, :(using RecurrentLayers); recursive=true)

makedocs(;
    modules=[RecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="RecurrentLayers.jl",
    format=Documenter.HTML(;
        canonical="https://MartinuzziFrancesco.github.io/RecurrentLayers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/RecurrentLayers.jl",
    devbranch="main",
)
