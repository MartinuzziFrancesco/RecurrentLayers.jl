<p align="center">
    <img width="400px" src="docs/src/assets/logo.png"/>
</p>

<div align="center">


| **Documentation** | **Build Status** | **Julia** | **Testing** |
|:-----------------:|:----------------:|:---------:|:-----------:|
| [![docs][docs-img]][docs-url] | [![CI][ci-img]][ci-url] [![codecov][cc-img]][cc-url] | [![Julia][julia-img]][julia-url] [![Code Style: Blue][style-img]][style-url] | [![Aqua QA][aqua-img]][aqua-url] [![JET][jet-img]][jet-url] |

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/dev/

[ci-img]: https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml?query=branch%3Amain

[cc-img]: https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl/branch/main/graph/badge.svg
[cc-url]: https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl

[julia-img]: https://img.shields.io/badge/julia-v1.10+-blue.svg
[julia-url]: https://julialang.org/

[style-img]: https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826
[style-url]: https://github.com/SciML/SciMLStyle

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[jet-img]: https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red
[jet-url]: https://github.com/aviatesk/JET.jl


</div>

<div align="center">
    <h2>RecurrentLayers.jl</h2>
</div>

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) recurrent layers offering by providing implementations of bleeding edge recurrent layers not commonly available in base deep learning libraries. It is designed for a seamless integration with the larger Flux ecosystem, enabling researchers and practitioners to leverage the latest developments in recurrent neural networks.

## Features 🚀

The package offers multiple cells and architectures:
 - Modifications of vanilla RNNs:
   [Independently recurrent neural networks](https://arxiv.org/abs/1803.04831),
   [Structurally constrained recurrent neural network](https://arxiv.org/pdf/1412.7753), and
   [FastRNN](https://arxiv.org/pdf/1901.02358)

 - Variations over gated architectures:
   [Minimal gated unit](https://arxiv.org/abs/1603.09420),
   [Light gated recurrent networks](https://arxiv.org/abs/1803.10225),
   [Recurrent addictive networks](https://arxiv.org/abs/1705.07393),
   [Light recurrent networks](https://www.mdpi.com/2079-9292/13/16/3204),
   [Neural architecture search networks](https://arxiv.org/abs/1611.01578),
   [Evolving recurrent neural networks](https://proceedings.mlr.press/v37/jozefowicz15.pdf),
   [Peephole long short term memory](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf), and
   [FastGRNN](https://arxiv.org/pdf/1901.02358)

 - Discretized ordinary differential equation formulations of RNNs:
   [Long expressive memory networks](https://arxiv.org/pdf/2110.04744), 
   [Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951), and
   [Antisymmetric recurrent neural network](https://arxiv.org/abs/1902.09689)

 - Additional more complex architectures:
   [Recurrent highway networks](https://arxiv.org/pdf/1607.03474),
   and [FastSlow RNNs](https://arxiv.org/abs/1705.08639)

 - Additional wrappers: [Stacked RNNs](https://arxiv.org/pdf/1312.6026)



## Installation 💻

You can install `RecurrentLayers` using either of:

```julia
using Pkg
Pkg.add("RecurrentLayers")
```

```julia_repl
julia> ]
pkg> add RecurrentLayers
```

## Getting started 🛠️

The workflow is identical to any recurrent Flux layer: just plug in a new recurrent layer in your workflow and test it out!

## License 📜

This project is licensed under the MIT License, except for `nas_cell.jl`, which is licensed under the Apache License, Version 2.0.

- `nas_cell.jl` is a reimplementation of the NASCell from TensorFlow and is licensed under the Apache License 2.0. See the file header and `LICENSE-APACHE` for details.
- All other files are licensed under the MIT License. See `LICENSE-MIT` for details.


## Support 🆘

If you have any questions, issues, or feature requests, please open an issue or contact us via email.