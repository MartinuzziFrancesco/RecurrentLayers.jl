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
RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) recurrent layers offering by providing implementations of bleeding edge recurrent layers not commonly available in base deep learning libraries. It is design for a seamless integration with the larger Flux ecosystem, enabling researchers and practitioners to leverage the latest developments in recurrent networks.

## Features

Currently available layers and work in progress in the short term:
 - [x] Minimal gated unit (MGU) [arxiv](https://arxiv.org/abs/1603.09420)
 - [x] Light gated recurrent unit (LiGRU) [arxiv](https://arxiv.org/abs/1803.10225)
 - [ ] Minimal gated recurrent unit (minGRU) and minimal long short term memory [arxiv](https://arxiv.org/abs/2410.01201)
 - [ ] Independently recurrent neural networks (IndRNN) [arxiv](https://arxiv.org/abs/1803.04831)

## Installation

RecurrentLayers.jl is not yet registered. You can install it directly from the GitHub repository:
```julia
using Pkg
Pkg.add(url="https://github.com/MartinuzziFrancesco/RecurrentLayers.jl")
```

## Getting started (to test!)

The workflow is identical to any recurrent Flux layer:

```julia
using Flux
using RecurrentLayers

# Define the model
model = Chain(
    Recur(MGU(input_size, hidden_size)),
    Dense(hidden_size, output_size),
    softmax
)

# Dummy data
x = rand(Float32, input_size, sequence_length, batch_size)
y = rand(1:output_size, batch_size)

# Define loss and optimizer
loss_fn = Flux.crossentropy
opt = ADAM()

# Training loop
for epoch in 1:10
    Flux.train!(loss_fn, params(model)) do
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
    end
end
```
