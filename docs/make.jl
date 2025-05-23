using RecurrentLayers, Flux, Documenter, DocumenterInterLinks
include("pages.jl")

mathengine = Documenter.MathJax()

links = InterLinks(
    "Flux" => "https://fluxml.ai/Flux.jl/stable/",
)

makedocs(;
    modules=[RecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="RecurrentLayers.jl",
    clean=true, doctest=true,
    linkcheck=true,
    format=Documenter.HTML(;
        mathengine,
        assets=["assets/favicon.ico"],
        canonical="https://MartinuzziFrancesco.github.io/RecurrentLayers.jl",
        edit_link="main"
    ),
    pages=pages,
    plugins=[links]
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/RecurrentLayers.jl",
    devbranch="main",
    push_preview=true
)
