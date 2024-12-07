using RecurrentLayers
using Documenter
include("pages.jl")

DocMeta.setdocmeta!(RecurrentLayers, :DocTestSetup, :(using RecurrentLayers); recursive=true)
mathengine = Documenter.MathJax()

makedocs(;
    modules=[RecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="RecurrentLayers.jl",
    format=Documenter.HTML(;
        mathengine,
        assets = ["assets/favicon.ico"],
        canonical="https://MartinuzziFrancesco.github.io/RecurrentLayers.jl",
        edit_link="main",
    ),
    pages=pages,
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/RecurrentLayers.jl",
    devbranch="main",
    push_preview=true,
)
