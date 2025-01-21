```@meta
CurrentModule = RecurrentLayers
```

# RecurrentLayers

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) recurrent layers offering by providing implementations of bleeding edge recurrent layers not commonly available in base deep learning libraries. It is designed for a seamless integration with the larger Flux ecosystem, enabling researchers and practitioners to leverage the latest developments in recurrent neural networks.

## Implemented layers

Cells and layers:
 - [x] Minimal gated unit (MGU) [arxiv](https://arxiv.org/abs/1603.09420)
 - [x] Light gated recurrent unit (LiGRU) [arxiv](https://arxiv.org/abs/1803.10225)
 - [x] Independently recurrent neural networks (IndRNN) [arxiv](https://arxiv.org/abs/1803.04831)
 - [x] Recurrent addictive networks (RAN) [arxiv](https://arxiv.org/abs/1705.07393)
 - [x] Recurrent highway network (RHN) [arixv](https://arxiv.org/pdf/1607.03474)
 - [x] Light recurrent unit (LightRU) [pub](https://www.mdpi.com/2079-9292/13/16/3204)
 - [x] Neural architecture search unit (NAS) [arxiv](https://arxiv.org/abs/1611.01578)
 - [x] Evolving recurrent neural networks (MUT1/2/3) [pub](https://proceedings.mlr.press/v37/jozefowicz15.pdf)
 - [x] Structurally constrained recurrent neural network (SCRN) [arxiv](https://arxiv.org/pdf/1412.7753)
 - [x] Peephole long short term memory (PeepholeLSTM) [pub](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf)
 - [x] FastRNN and FastGRNN [arxiv](https://arxiv.org/pdf/1901.02358)

Wrappers:
 - [x] Stacked RNNs
 - [x] FastSlow RNNs [arxiv](https://arxiv.org/abs/1705.08639)

## Contributing

Contributions are always welcome! We specifically look for :
 - Recurrent cells you would like to see implemented 
 - Benchmarks
 - Fixes for any bugs/errors
 - Documentation, in any form: examples, how tos, docstrings  