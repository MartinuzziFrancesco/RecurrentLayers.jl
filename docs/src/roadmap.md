# Roadmap
This page documents some planned work for RecurrentLayers.jl.
Future work for this library includes additional cells such as:

 - FastRNNs and FastGRUs (current focus) [arxiv](https://arxiv.org/abs/1901.02358)
 - Unitary recurrent neural networks [arxiv](https://arxiv.org/abs/1611.00035)
 - Modern recurrent neural networks such as [LRU](https://arxiv.org/abs/2303.06349) 
   and [minLSTM/minGRU](https://arxiv.org/abs/2410.01201)
 - Quasi recurrent neural networks [arxiv](https://arxiv.org/abs/1611.01576)

Additionally, some cell-independent architectures are also planned,
that expand the ability of recurrent architectures and could theoretically take
any cell:

 - Clockwork rnns [arxiv](https://arxiv.org/abs/1402.3511)
 - Phased rnns [arxiv](https://arxiv.org/abs/1610.09513)
 - Segment rnn [arxiv](https://arxiv.org/abs/2308.11200)
 - Fast-Slow rnns [arxiv](https://arxiv.org/abs/1705.08639)

An implementation of these ideally would be,
for example `FastSlow(RNNCell, input_size => hidden_size)`.
More details on this soon!