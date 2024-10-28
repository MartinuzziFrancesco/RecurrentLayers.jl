# RecurrentLayers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/dev/)
[![Build Status](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

> [!CAUTION]
> Currently still under HEAVY development. Use at own risk and beware.

> [!WARNING]  
> Tests and benchmark are still missing. Layers may not work as intended yet.


## Overview
RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) recurrent layers offering by providing implementations of bleeding edge recurrent layers not commonly available in base deep learning libraries. It is designed for a seamless integration with the larger Flux ecosystem, enabling researchers and practitioners to leverage the latest developments in recurrent neural networks.

## Features

Currently available layers and work in progress in the short term:
 - [x] Minimal gated unit (MGU) [arxiv](https://arxiv.org/abs/1603.09420)
 - [x] Light gated recurrent unit (LiGRU) [arxiv](https://arxiv.org/abs/1803.10225)
 - [x] Independently recurrent neural networks (IndRNN) [arxiv](https://arxiv.org/abs/1803.04831)
 - [x] Recurrent addictive networks (RAN) [arxiv](https://arxiv.org/abs/1705.07393)
 - [x] Recurrent highway network (RHN) [arixv](https://arxiv.org/pdf/1607.03474)
  - [x] Light recurrent unit (LightRU) [pub](https://www.mdpi.com/2079-9292/13/16/3204)
  - [ ] Minimal gated recurrent unit (minGRU) and minimal long short term memory (minLSTM) [arxiv](https://arxiv.org/abs/2410.01201)

## Installation

RecurrentLayers.jl is not yet registered. You can install it directly from the GitHub repository:
```julia
using Pkg
Pkg.add(url="https://github.com/MartinuzziFrancesco/RecurrentLayers.jl")
```

## Getting started

The workflow is identical to any recurrent Flux layer:

```julia
using Flux
using RecurrentLayers

input_size = 2
hidden_size = 5
output_size = 3
sequence_length = 100
epochs = 10

model = Chain(
    MGU(input_size, hidden_size),
    Dense(hidden_size, output_size)
)

# dummy data
X = rand(Float32, input_size, sequence_length)
Y = rand(1:output_size)

# loss function
loss_fn(x, y) = Flux.mse(model(x), y)

# optimizer
opt = Adam()

# training 
for epoch in 1:epochs
    # gradients
    gs = gradient(Flux.params(model)) do
        loss = loss_fn(X, Y)
        return loss
    end
    # update parameters
    Flux.update!(opt, Flux.params(model), gs)
    # loss at epoch
    current_loss = loss_fn(X, Y)
    println("Epoch $epoch, Loss: $(current_loss)")
end
```
