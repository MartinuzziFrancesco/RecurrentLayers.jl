```@meta
CurrentModule = RecurrentLayers
```

# RecurrentLayers

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) recurrent layers offering by providing implementations of bleeding edge recurrent layers not commonly available in base deep learning libraries. It is designed for a seamless integration with the larger Flux ecosystem, enabling researchers and practitioners to leverage the latest developments in recurrent neural networks.

## Implemented layers


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
   with its gated version

 - Additional more complex architectures:
   [Recurrent highway networks](https://arxiv.org/pdf/1607.03474),
   and [FastSlow RNNs](https://arxiv.org/abs/1705.08639)

 - Additional wrappers: [Stacked RNNs](https://arxiv.org/pdf/1312.6026)

## Contributing

Contributions are always welcome! We specifically look for :
 - Recurrent cells you would like to see implemented 
 - Benchmarks
 - Fixes for any bugs/errors
 - Documentation, in any form: examples, how tos, docstrings  