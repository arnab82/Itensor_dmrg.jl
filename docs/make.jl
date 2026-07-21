using Documenter
using NaiveDMRG

makedocs(
    sitename = "NaiveDMRG.jl",
    modules = [NaiveDMRG],
    authors = "arnab82",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://arnab82.github.io/NaiveDMRG.jl",
        mathengine = Documenter.KaTeX(),
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Theory" => "theory.md",
        "Implementation" => "implementation.md",
        "API reference" => "api.md",
    ],
    # A first, resilient deploy: don't fail the build on undocumented exports or
    # docstring @ref links that don't resolve to a page.
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/arnab82/NaiveDMRG.jl.git",
    devbranch = "main",
    push_preview = true,
)
