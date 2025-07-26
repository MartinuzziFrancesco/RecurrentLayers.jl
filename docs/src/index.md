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
| **LightRU** | MDPI Electronics 2023 | – |
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
| [**UGRNN**](https://arxiv.org/abs/1611.09913) | ICLR 2017 | - |
| [**UnICORNN**](https://arxiv.org/abs/2103.05487) | ICML 2021 | [tk-rusch/unicornn](https://github.com/tk-rusch/unicornn) |
| [**WMCLSTM**](https://arxiv.org/abs/2109.00020) | Neural Networks 2021 | – |

 - Additional wrappers: [Stacked RNNs](https://arxiv.org/pdf/1312.6026), and [Multiplicative RNN](https://icml.cc/2011/papers/524_icmlpaper.pdf).

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
