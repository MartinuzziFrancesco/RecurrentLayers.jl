using Documenter, DocumenterCitations, DocumenterInterLinks,
      RecurrentLayers, Flux
include("pages.jl")

mathengine = Documenter.MathJax()

links = InterLinks(
    "Flux" => "https://fluxml.ai/Flux.jl/stable/",
)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :authoryear
)

makedocs(;
    modules=[RecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="RecurrentLayers.jl",
    clean=true, doctest=true,
    linkcheck=true,
    plugins=[links, bib],
    format=Documenter.HTML(;
        mathengine,
        assets=["assets/favicon.ico"],
        canonical="https://MartinuzziFrancesco.github.io/RecurrentLayers.jl",
        edit_link="main"
    ),
    pages=pages,
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/RecurrentLayers.jl",
    devbranch="main",
    push_preview=true
)
