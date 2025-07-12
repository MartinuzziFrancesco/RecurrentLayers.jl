<p align="center">
    <img width="400px" src="docs/src/assets/logo.png"/>
</p>

<div align="center">


| **Documentation** | **Build Status** | **Julia** | **Testing** |
|:-----------------:|:----------------:|:---------:|:-----------:|
| [![docsstbl][docs-stbl]][docsstbl-url] [![docsdev][docs-dev]][docsdev-url] | [![CI][ci-img]][ci-url] | [![Julia][julia-img]][julia-url] [![Code Style: Blue][style-img]][style-url] | [![Aqua QA][aqua-img]][aqua-url] [![JET][jet-img]][jet-url] [![codecov][cc-img]][cc-url] |

[docs-stbl]: https://img.shields.io/badge/docs-stable-blue.svg
[docsstbl-url]: https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/stable/

[docs-dev]: https://img.shields.io/badge/docs-dev-blue.svg
[docsdev-url]: https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/dev/

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

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl)
recurrent layers offering by providing implementations of additional
recurrent layers not available in base deep learning libraries.

## Features 🚀

The package offers multiple layers for [Flux.jl](https://github.com/FluxML/Flux.jl).
Currently there are 30+ cells implemented, together with multiple higher
level implementations:

| Short name | Publication venue | Official implementation |
|------------|-------------------|-----------------------------|
| [**AntisymmetricRNN/GatedAntisymmetricRNN**](https://arxiv.org/abs/1902.09689) | ICLR 2019 | – |
| [**ATR**](https://arxiv.org/abs/1810.12546) | EMNLP 2018 | [bzhangGo/ATR](https://github.com/bzhangGo/ATR) |
| [**BR/BRC**](https://doi.org/10.1371/journal.pone.0252676) | PLOS ONE 2021 | [nvecoven/BRC](https://github.com/nvecoven/BRC) |
| [**CFN**](https://arxiv.org/abs/1612.06212) | ICLR 2017 | – |
| [**coRNN**](https://arxiv.org/abs/2010.00951) | ICLR 2021 | [tk-rusch/coRNN](https://github.com/tk-rusch/coRNN) |
| [**FastRNN/FastGRNN**](https://arxiv.org/abs/1901.02358) | NeurIPS 2018 | [Microsoft/EdgeML](https://github.com/Microsoft/EdgeML) |
| [**FSRNN**](https://arxiv.org/abs/1705.08639) | NeurIPS 2017 | [amujika/Fast-Slow-LSTM](https://github.com/amujika/Fast-Slow-LSTM) |
| [**IndRNN**](https://arxiv.org/abs/1803.04831) | CVPR 2018 | [Sunnydreamrain/IndRNN_Theano_Lasagne](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne) |
| [**JANET**](https://arxiv.org/abs/1804.04849) | arXiv 2018 | [JosvanderWesthuizen/janet](https://github.com/JosvanderWesthuizen/janet) |
| [**LEM**](https://arxiv.org/pdf/2110.04744) | ICLR 2022 | [tk-rusch/LEM](https://github.com/tk-rusch/LEM) |
| [**LiGRU**](https://arxiv.org/abs/1803.10225) | IEEE Transactions on Emerging Topics in Computing 2018 | [mravanelli/theano-kaldi-rnn](https://github.com/mravanelli/theano-kaldi-rnn/) |
| [**LightRU**](https://www.mdpi.com/2079-9292/13/16/3204) | MDPI Electronics 2023 | – |
| [**MinimalRNN**](https://arxiv.org/abs/1711.06788) | NeurIPS 2017 | – |
| [**MultiplicativeLSTM**](https://arxiv.org/abs/1609.07959) | Workshop ICLR 2017 | [benkrause/mLSTM](https://github.com/benkrause/mLSTM) |
| [**MGU**](https://arxiv.org/abs/1603.09420) | International Journal of Automation and Computing 2016 | – |
| [**MUT1/MUT2/MUT3**](https://proceedings.mlr.press/v37/jozefowicz15.pdf) | ICML 2015 | – |
| [**NAS**](https://arxiv.org/abs/1611.01578) | arXiv 2016 | [tensorflow_addons/rnn](https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/rnn/nas_cell.py#L29-L236) |
| [**OriginalLSTM**](https://ieeexplore.ieee.org/abstract/document/6795963) | Neural Computation 1997 | - |
| [**PeepholeLSTM**](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf) | JMLR 2002 | – |
| [**RAN**](https://arxiv.org/abs/1705.07393) | arXiv 2017 | [kentonl/ran](https://github.com/kentonl/ran) |
| [**RHN**](https://arxiv.org/abs/1607.03474) | ICML 2017 | [jzilly/RecurrentHighwayNetworks](https://github.com/jzilly/RecurrentHighwayNetworks) |
| [**SCRN**](https://arxiv.org/abs/1412.7753) | ICLR 2015 | [facebookarchive/SCRNNs](https://github.com/facebookarchive/SCRNNs) |
| [**SGRN**](https://doi.org/10.1049/gtd2.12056) | IET 2018 | – |
| [**STAR**](https://arxiv.org/abs/1911.11033) | IEEE Transactions on Pattern Analysis and Machine Intelligence 2022 | [0zgur0/STAckable-Recurrent-network](https://github.com/0zgur0/STAckable-Recurrent-network) |
| [**Typed RNN / GRU / LSTM**](https://arxiv.org/abs/1602.02218) | ICML 2016 | – |
| [**UnICORNN**](https://arxiv.org/abs/2103.05487) | ICML 2021 | [tk-rusch/unicornn](https://github.com/tk-rusch/unicornn) |
| [**WMCLSTM**](https://arxiv.org/abs/2109.00020) | Neural Networks 2021 | – |

 - Additional wrappers: [Stacked RNNs](https://arxiv.org/abs/1312.6026), and
 [Multiplicative RNN](https://icml.cc/2011/papers/524_icmlpaper.pdf).



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


## See also

[LuxRecurrentLayers.jl](https://github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl):
Equivalent library, providing recurrent layers for Lux.jl.

[torchrecurrent](https://github.com/MartinuzziFrancesco/torchrecurrent):
Recurrent layers for Pytorch.

[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl):
Reservoir computing utilities for scientific machine learning.
Essentially gradient free trained neural networks.
