# This file is a reimplementation in Julia of the NASCell as described in:
# "Neural Architecture Search with Reinforcement Learning" (https://arxiv.org/pdf/1611.01578).
# The original implementation in TensorFlow can be found here:
# https://www.tensorflow.org/addons/api_docs/python/tfa/rnn/NASCell
# No changes were made that alter the behavior of the cell compared to the original
# implementation; differences may be due to language-specific syntax.
#
# The original implementation is licensed under the Apache License, Version 2.0.
# This reimplementation is also licensed under the Apache License, Version 2.0.

#
# Copyright 2024 Francesco Martinuzzi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

struct NASCell{I,H,V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer NASCell

@doc raw"""
    NASCell((input_size => hidden_size);
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Neural Architecture Search unit](https://arxiv.org/pdf/1611.01578).
See [`NAS`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\text{First Layer Outputs:} & \\
o_1 &= \sigma(W_i^{(1)} x_t + W_h^{(1)} h_{t-1} + b^{(1)}), \\
o_2 &= \text{ReLU}(W_i^{(2)} x_t + W_h^{(2)} h_{t-1} + b^{(2)}), \\
o_3 &= \sigma(W_i^{(3)} x_t + W_h^{(3)} h_{t-1} + b^{(3)}), \\
o_4 &= \text{ReLU}(W_i^{(4)} x_t \cdot W_h^{(4)} h_{t-1}), \\
o_5 &= \tanh(W_i^{(5)} x_t + W_h^{(5)} h_{t-1} + b^{(5)}), \\
o_6 &= \sigma(W_i^{(6)} x_t + W_h^{(6)} h_{t-1} + b^{(6)}), \\
o_7 &= \tanh(W_i^{(7)} x_t + W_h^{(7)} h_{t-1} + b^{(7)}), \\
o_8 &= \sigma(W_i^{(8)} x_t + W_h^{(8)} h_{t-1} + b^{(8)}). \\

\text{Second Layer Computations:} & \\
l_1 &= \tanh(o_1 \cdot o_2) \\
l_2 &= \tanh(o_3 + o_4) \\
l_3 &= \tanh(o_5 \cdot o_6) \\
l_4 &= \sigma(o_7 + o_8) \\

\text{Inject Cell State:} & \\
l_1 &= \tanh(l_1 + c_{\text{state}}) \\

\text{Final Layer Computations:} & \\
c_{\text{new}} &= l_1 \cdot l_2 \\
l_5 &= \tanh(l_3 + l_4) \\
h_{\text{new}} &= \tanh(c_{\text{new}} \cdot l_5)
\end{aligned}
```

# Forward

    rnncell(inp, [state])
"""
function NASCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)
    Wi = init_kernel(8 * hidden_size, input_size)
    Wh = init_recurrent_kernel(8 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return NASCell(Wi, Wh, b)
end

function (nas::NASCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(nas, inp, 1 => size(nas.Wi,2))
    Wi, Wh, b = nas.Wi, nas.Wh, nas.bias

    #matmul and split
    im = chunk(Wi * inp, 8; dims=1)
    mm = chunk(Wh * state .+ b, 8; dims=1)

    #first layer
    layer1_1 = sigmoid_fast(im[1] .+ mm[1])
    layer1_2 = relu(im[2] .+ mm[2])
    layer1_3 = sigmoid_fast(im[3] .+ mm[3])
    layer1_4 = relu(im[4] .* mm[4])
    layer1_5 = tanh_fast(im[5] .+ mm[5])
    layer1_6 = sigmoid_fast(im[6] .+ mm[6])
    layer1_7 = tanh_fast(im[7] .+ mm[7])
    layer1_8 = sigmoid_fast(im[8] .+ mm[8])

    #second layer
    l2_1 = tanh_fast(layer1_1 .* layer1_2)
    l2_2 = tanh_fast(layer1_3 .+ layer1_4)
    l2_3 = tanh_fast(layer1_5 .* layer1_6)
    l2_4 = sigmoid_fast(layer1_7 .+ layer1_8)

    #inject cell
    l2_1 = tanh_fast(l2_1 .+ c_state)

    # Third layer
    new_cstate = l2_1 .* l2_2
    l3_2 = tanh_fast(l2_3 .+ l2_4)

    new_state = tanh_fast(new_cstate .* l3_2)

    return new_state, (new_state, new_cstate)
end

Base.show(io::IO, nas::NASCell) =
    print(io, "NASCell(", size(nas.Wi, 2), " => ", size(nas.Wi, 1)÷8, ")")


struct NAS{M} <: AbstractRecurrentLayer
    cell::M
end

Flux.@layer :expand NAS

@doc raw"""
    NAS((input_size => hidden_size)::Pair; kwargs...)


[Neural Architecture Search unit](https://arxiv.org/pdf/1611.01578).
See [`NASCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\text{First Layer Outputs:} & \\
o_1 &= \sigma(W_i^{(1)} x_t + W_h^{(1)} h_{t-1} + b^{(1)}), \\
o_2 &= \text{ReLU}(W_i^{(2)} x_t + W_h^{(2)} h_{t-1} + b^{(2)}), \\
o_3 &= \sigma(W_i^{(3)} x_t + W_h^{(3)} h_{t-1} + b^{(3)}), \\
o_4 &= \text{ReLU}(W_i^{(4)} x_t \cdot W_h^{(4)} h_{t-1}), \\
o_5 &= \tanh(W_i^{(5)} x_t + W_h^{(5)} h_{t-1} + b^{(5)}), \\
o_6 &= \sigma(W_i^{(6)} x_t + W_h^{(6)} h_{t-1} + b^{(6)}), \\
o_7 &= \tanh(W_i^{(7)} x_t + W_h^{(7)} h_{t-1} + b^{(7)}), \\
o_8 &= \sigma(W_i^{(8)} x_t + W_h^{(8)} h_{t-1} + b^{(8)}). \\

\text{Second Layer Computations:} & \\
l_1 &= \tanh(o_1 \cdot o_2) \\
l_2 &= \tanh(o_3 + o_4) \\
l_3 &= \tanh(o_5 \cdot o_6) \\
l_4 &= \sigma(o_7 + o_8) \\

\text{Inject Cell State:} & \\
l_1 &= \tanh(l_1 + c_{\text{state}}) \\

\text{Final Layer Computations:} & \\
c_{\text{new}} &= l_1 \cdot l_2 \\
l_5 &= \tanh(l_3 + l_4) \\
h_{\text{new}} &= \tanh(c_{\text{new}} \cdot l_5)
\end{aligned}
```
"""
function NAS((input_size, hidden_size)::Pair; kwargs...)
    cell = NASCell(input_size => hidden_size; kwargs...)
    return NAS(cell)
end

function (nas::NAS)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    return scan(nas.cell, inp, state)
end
