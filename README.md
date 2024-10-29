# RecurrentLayers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/dev/)
[![Build Status](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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
 - [x] Neural architecture search unit (NAS) [arxiv](https://arxiv.org/abs/1611.01578)
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
using RecurrentLayers

using Flux
using MLUtils: DataLoader
using Statistics
using Random

# Parameters
input_size = 1       # Each element in the sequence is a scalar
hidden_size = 64     # Size of the hidden state in MGU
num_classes = 2      # Binary classification
seq_length = 10      # Length of each sequence
batch_size = 16      # Batch size
num_epochs = 50       # Number of epochs for training
num_samples = 1000   # Number of samples in dataset

# Create dataset
function create_dataset(seq_length, num_samples)
    data = randn(input_size, seq_length, num_samples)
    labels = sum(data, dims=(1,2)) .>= 0
    labels = Int.(labels)
    return data, labels
end

# Generate training data
train_data, train_labels = create_dataset(seq_length, num_samples)
train_loader = DataLoader((train_data, train_labels), batchsize=batch_size, shuffle=true)

# Define the model
model = Chain(
    RAN(input_size => hidden_size),
    x -> x[:, end, :],  # Extract the last hidden state
    Dense(hidden_size, num_classes)
)

function adjust_labels(labels)
    return labels .+ 1
end

# Define the loss function
function loss_fn(batch_data, batch_labels)
    # Adjust labels
    batch_labels = adjust_labels(batch_labels)
    # One-hot encode labels and remove any extra singleton dimensions
    batch_labels_oh = dropdims(Flux.onehotbatch(batch_labels, 1:num_classes), dims=(2, 3))
    # Forward pass
    y_pred = model(batch_data)
    # Compute loss
    loss = Flux.logitcrossentropy(y_pred, batch_labels_oh)
    return loss
end


# Define the optimizer
opt = Adam(0.01)

# Training loop
for epoch in 1:num_epochs
    total_loss = 0.0
    for (batch_data, batch_labels) in train_loader
        # Compute gradients and update parameters
        grads = gradient(() -> loss_fn(batch_data, batch_labels), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), grads)

        # Accumulate loss
        total_loss += loss_fn(batch_data, batch_labels)
    end
    avg_loss = total_loss / length(train_loader)
    println("Epoch $epoch/$num_epochs, Loss: $(round(avg_loss, digits=4))")
end

# Generate test data
test_data, test_labels = create_dataset(seq_length, 200)
test_loader = DataLoader((test_data, test_labels), batchsize=batch_size, shuffle=false)

# Evaluation
correct = 0
total = 0
for (batch_data, batch_labels) in test_loader
    # Adjust labels
    batch_labels = adjust_labels(batch_labels)
    # Forward pass
    y_pred = model(batch_data)
    # Decode predictions
    predicted = Flux.onecold(y_pred, 1:num_classes)
    # Flatten and compare
    correct += sum(vec(predicted) .== vec(batch_labels))
    total += length(batch_labels)
end

accuracy = 100 * correct / total
println("Test Accuracy: $(round(accuracy, digits=2))%")


```
## License

This project is licensed under the MIT License, except for `nas_cell.jl`, which is licensed under the Apache License, Version 2.0.

- `nas_cell.jl` is a reimplementation of the NASCell from TensorFlow and is licensed under the Apache License 2.0. See the file header and `LICENSE-APACHE` for details.
- All other files are licensed under the MIT License. See `LICENSE-MIT` for details.
