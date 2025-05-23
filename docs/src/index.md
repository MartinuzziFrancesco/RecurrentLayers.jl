```@meta
CurrentModule = RecurrentLayers
```

# RecurrentLayers

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl)
recurrent layers offering by providing implementations of additional
recurrent layers not available in base deep learning libraries.

## Features

The package offers multiple layers for [Flux.jl](https://github.com/FluxML/Flux.jl).
Currently there are 30+ cells implemented, together with multiple higher
level implementations:
 - Modifications of vanilla RNNs:
   [Independently recurrent neural networks](https://arxiv.org/abs/1803.04831),
   [Structurally constrained recurrent neural network](https://arxiv.org/pdf/1412.7753),
   [FastRNN](https://arxiv.org/pdf/1901.02358), and
   [Typed RNNs](https://arxiv.org/abs/1602.02218).

 - Variations over gated architectures:
   [Minimal gated unit](https://arxiv.org/abs/1603.09420),
   [Light gated recurrent networks](https://arxiv.org/abs/1803.10225),
   [Recurrent addictive networks](https://arxiv.org/abs/1705.07393),
   [Light recurrent networks](https://www.mdpi.com/2079-9292/13/16/3204),
   [Neural architecture search networks](https://arxiv.org/abs/1611.01578),
   [Evolving recurrent neural networks](https://proceedings.mlr.press/v37/jozefowicz15.pdf),
   [Peephole long short term memory](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf),
   [FastGRNN](https://arxiv.org/pdf/1901.02358),
   [Just another network](https://arxiv.org/abs/1804.04849),
   [Chaos free network](https://arxiv.org/abs/1612.06212),
   [Typed gated recurrent unit](https://arxiv.org/abs/1602.02218),
   [Typed long short term memory](https://arxiv.org/abs/1602.02218),
   [Stackable recurrent network](https://arxiv.org/abs/1911.11033),
   [Minimal recurrent neural network](https://arxiv.org/abs/1711.06788),
   [Addition-subtraction twin-gated recurrent cell](https://arxiv.org/abs/1810.12546),
   [Simple gated recurrent network](https://doi.org/10.1049/gtd2.12056),
   [Bistable recurrent cell](https://doi.org/10.1371/journal.pone.0252676),
   [Recurrently neuromodulated bistable recurrent cell](https://doi.org/10.1371/journal.pone.0252676),
   [Peephole long short term memory cell](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf), and
   [Long short term memory cell with working memory connections](https://arxiv.org/abs/2109.00020).

 - Discretized ordinary differential equation formulations of RNNs:
   [Long expressive memory networks](https://arxiv.org/pdf/2110.04744), 
   [Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951),
   [Antisymmetric recurrent neural network](https://arxiv.org/abs/1902.09689)
   with its gated version, and
   [Undamped independent controlled oscillatory recurrent neural network](https://arxiv.org/abs/2010.00951).

 - Additional more complex architectures:
   [Recurrent highway networks](https://arxiv.org/pdf/1607.03474),
   and [FastSlow RNNs](https://arxiv.org/abs/1705.08639)

 - Additional wrappers: [Stacked RNNs](https://arxiv.org/pdf/1312.6026), and
 [Multiplicative RNN](https://icml.cc/2011/papers/524_icmlpaper.pdf).

## Installation

You can install `RecurrentLayers` using either of:

```julia
using Pkg
Pkg.add("RecurrentLayers")
```

```julia_repl
julia> ]
pkg> add RecurrentLayers
```

## Contributing

Contributions are always welcome! We specifically look for :
 - Recurrent cells you would like to see implemented 
 - Benchmarks
 - Fixes for any bugs/errors
 - Documentation, in any form: examples, how tos, docstrings

Please consider the following guidelines before opening a pull request:
 - The code should be formatted according to the format file provided
 - Variable names should be meaningful: please no single letter variables,
   and try to avoid double letters variables too. I know at the moment there are
   some in the codebase, but I will need a breaking change in order to fix the majority of them.
 - The format file does not format markdown. If you are adding docs, or docstrings
   please take care of not going over 92 cols.

For any clarification feel free to contact me directly (@MartinuzziFrancesco)
either in the julia slack, by email or X/bluesky.