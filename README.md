<p align="center">
    <img width="400px" src="docs/src/assets/logo.png"/>
</p>

<div align="center">


| **Documentation** | **Build Status** | **Julia** | **Testing** |
|:-----------------:|:----------------:|:---------:|:---------|
| [![docs][docs-img]][docs-url] | [![CI][ci-img]][ci-url] [![codecov][cc-img]][cc-url] | [![Julia][julia-img]][julia-url] [![Code Style: Blue][style-img]][style-url] | [![Aqua QA][aqua-img]][aqua-url] |

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://MartinuzziFrancesco.github.io/RecurrentLayers.jl/dev/

[ci-img]: https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/MartinuzziFrancesco/RecurrentLayers.jl/actions/workflows/CI.yml?query=branch%3Amain

[cc-img]: https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl/branch/main/graph/badge.svg
[cc-url]: https://codecov.io/gh/MartinuzziFrancesco/RecurrentLayers.jl

[julia-img]: https://img.shields.io/badge/julia-v1.10+-blue.svg
[julia-url]: https://julialang.org/

[style-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[style-url]: https://github.com/invenia/BlueStyle

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

Currently available layers and work in progress in the short term:
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
 - [ ] Minimal gated recurrent unit (minGRU) and minimal long short term memory (minLSTM) [arxiv](https://arxiv.org/abs/2410.01201)

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

The workflow is identical to any recurrent Flux layer:

```julia
using RecurrentLayers

using Flux
using MLUtils: DataLoader
using Statistics
using Random

# Create dataset
function create_data(input_size, seq_length::Int, num_samples::Int)
    data = randn(input_size, seq_length, num_samples) #(input_size, seq_length, num_samples)
    labels = sum(data, dims=(1, 2)) .>= 0
    labels = Int.(labels)
    labels = dropdims(labels, dims=(1))
    return data, labels
end

function create_dataset(input_size, seq_length, n_train::Int, n_test::Int, batch_size)
    train_data, train_labels = create_data(input_size, seq_length, n_train)
    train_loader = DataLoader((train_data, train_labels), batchsize=batch_size, shuffle=true)

    test_data, test_labels = create_data(input_size, seq_length, n_test)
    test_loader = DataLoader((test_data, test_labels), batchsize=batch_size, shuffle=false)
    return train_loader, test_loader
end

struct RecurrentModel{H,C,D}
    h0::H
    rnn::C
    dense::D
end

Flux.@layer RecurrentModel trainable=(rnn, dense)

function RecurrentModel(input_size::Int, hidden_size::Int)
    return RecurrentModel(
                 zeros(Float32, hidden_size), 
                 MGU(input_size => hidden_size),
                 Dense(hidden_size => 1, sigmoid))
end

function (model::RecurrentModel)(inp)
    state = model.rnn(inp, model.h0)
    state = state[:, end, :]
    output = model.dense(state)
    return output
end

function criterion(model, batch_data, batch_labels)
    y_pred = model(batch_data)
    loss = Flux.binarycrossentropy(y_pred, batch_labels)
    return loss
end

function train_recurrent!(epoch, train_loader, opt, model, criterion)
    total_loss = 0.0
    for (batch_data, batch_labels) in train_loader
        # Compute gradients and update parameters
        grads = gradient(() -> criterion(model, batch_data, batch_labels), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), grads)

        # Accumulate loss
        total_loss += criterion(model, batch_data, batch_labels)
    end
    avg_loss = total_loss / length(train_loader)
    println("Epoch $epoch/$num_epochs, Loss: $(round(avg_loss, digits=4))")
end

function test_recurrent(test_loader, model)
    # Evaluation
    correct = 0
    total = 0
    for (batch_data, batch_labels) in test_loader

        # Forward pass
        predicted = model(batch_data)

        # Decode predictions: convert probabilities to class labels (0 or 1)
        predicted_labels = vec(predicted .>= 0.5)   # Threshold at 0.5 for binary classification

        # Compare predicted labels to actual labels
        correct += sum(predicted_labels .== vec(batch_labels))
        total += length(batch_labels)
    end
    accuracy = correct / total
    println("Accuracy: ", accuracy * 100, "%")
end

function main(;
    input_size = 1,       # Each element in the sequence is a scalar
    hidden_size = 64,    # Size of the hidden state
    seq_length = 10,      # Length of each sequence
    batch_size = 16,      # Batch size
    num_epochs = 50,       # Number of epochs for training
    n_train = 1000,   # Number of samples in train dataset
    n_test = 200   # Number of samples in test dataset)
)
    model = RecurrentModel(input_size, hidden_size)
    # Generate test data
    train_loader, test_loader = create_dataset(input_size, seq_length, n_train, n_test, batch_size)
    # Define the optimizer
    opt = Adam(0.001)

    for epoch in 1:num_epochs
        train_recurrent!(epoch, train_loader, opt, model, criterion)
    end

    test_recurrent(test_loader, model)

end

main()



```
## License 📜

This project is licensed under the MIT License, except for `nas_cell.jl`, which is licensed under the Apache License, Version 2.0.

- `nas_cell.jl` is a reimplementation of the NASCell from TensorFlow and is licensed under the Apache License 2.0. See the file header and `LICENSE-APACHE` for details.
- All other files are licensed under the MIT License. See `LICENSE-MIT` for details.


## Support 🆘

If you have any questions, issues, or feature requests, please open an issue or contact us via email.