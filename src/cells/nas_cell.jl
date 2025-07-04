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

@doc raw"""
    NASCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

Neural Architecture Search unit [Zoph2017](@cite).
See [`NAS`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{o}_1 &= \sigma\left( \mathbf{W}^{(1)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(1)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(1)} \right), \\
    \mathbf{o}_2 &= \text{ReLU}\left( \mathbf{W}^{(2)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(2)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(2)} \right), \\
    \mathbf{o}_3 &= \sigma\left( \mathbf{W}^{(3)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(3)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(3)} \right), \\
    \mathbf{o}_4 &= \text{ReLU}\left( \mathbf{W}^{(4)}_{ih} \mathbf{x}(t)
        \cdot \mathbf{W}^{(4)}_{hh} \mathbf{h}(t-1) \right), \\
    \mathbf{o}_5 &= \tanh\left( \mathbf{W}^{(5)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(5)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(5)} \right), \\
    \mathbf{o}_6 &= \sigma\left( \mathbf{W}^{(6)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(6)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(6)} \right), \\
    \mathbf{o}_7 &= \tanh\left( \mathbf{W}^{(7)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(7)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(7)} \right), \\
    \mathbf{o}_8 &= \sigma\left( \mathbf{W}^{(8)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(8)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(8)} \right), \\[1ex]
    \mathbf{l}_1 &= \tanh\left( \mathbf{o}_1 \cdot \mathbf{o}_2 \right), \\
    \mathbf{l}_2 &= \tanh\left( \mathbf{o}_3 + \mathbf{o}_4 \right), \\
    \mathbf{l}_3 &= \tanh\left( \mathbf{o}_5 \cdot \mathbf{o}_6 \right), \\
    \mathbf{l}_4 &= \sigma\left( \mathbf{o}_7 + \mathbf{o}_8 \right), \\[1ex]
    \mathbf{l}_1 &= \tanh\left( \mathbf{l}_1 + \mathbf{c}_{\text{state}}
        \right), \\[1ex]
    \mathbf{c}_{\text{new}} &= \mathbf{l}_1 \cdot \mathbf{l}_2, \\
    \mathbf{l}_5 &= \tanh\left( \mathbf{l}_3 + \mathbf{l}_4 \right), \\
    \mathbf{h}_{\text{new}} &= \tanh\left( \mathbf{c}_{\text{new}} \cdot
        \mathbf{l}_5 \right)
\end{aligned}
```

# Forward

    nascell(inp, (state, cstate))
    nascell(inp)

## Arguments

- `inp`: The input to the nascell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the NASCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct NASCell{I, H, V} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer NASCell

function NASCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(8 * hidden_size, input_size)
    Wh = init_recurrent_kernel(8 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return NASCell(Wi, Wh, b)
end

function (nas::NASCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(nas, inp, 1 => size(nas.Wi, 2))
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

function Base.show(io::IO, nas::NASCell)
    print(io, "NASCell(", size(nas.Wi, 2), " => ", size(nas.Wi, 1) ÷ 8, ")")
end

@doc raw"""
    NAS(input_size => hidden_size;
        return_state = false,
        kwargs...)


Neural Architecture Search unit [Zoph2017](@cite).
See [`NASCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{o}_1 &= \sigma\left( \mathbf{W}^{(1)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(1)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(1)} \right), \\
    \mathbf{o}_2 &= \text{ReLU}\left( \mathbf{W}^{(2)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(2)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(2)} \right), \\
    \mathbf{o}_3 &= \sigma\left( \mathbf{W}^{(3)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(3)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(3)} \right), \\
    \mathbf{o}_4 &= \text{ReLU}\left( \mathbf{W}^{(4)}_{ih} \mathbf{x}(t)
        \cdot \mathbf{W}^{(4)}_{hh} \mathbf{h}(t-1) \right), \\
    \mathbf{o}_5 &= \tanh\left( \mathbf{W}^{(5)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(5)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(5)} \right), \\
    \mathbf{o}_6 &= \sigma\left( \mathbf{W}^{(6)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(6)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(6)} \right), \\
    \mathbf{o}_7 &= \tanh\left( \mathbf{W}^{(7)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(7)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(7)} \right), \\
    \mathbf{o}_8 &= \sigma\left( \mathbf{W}^{(8)}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{(8)}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{(8)} \right), \\[1ex]
    \mathbf{l}_1 &= \tanh\left( \mathbf{o}_1 \cdot \mathbf{o}_2 \right), \\
    \mathbf{l}_2 &= \tanh\left( \mathbf{o}_3 + \mathbf{o}_4 \right), \\
    \mathbf{l}_3 &= \tanh\left( \mathbf{o}_5 \cdot \mathbf{o}_6 \right), \\
    \mathbf{l}_4 &= \sigma\left( \mathbf{o}_7 + \mathbf{o}_8 \right), \\[1ex]
    \mathbf{l}_1 &= \tanh\left( \mathbf{l}_1 + \mathbf{c}_{\text{state}}
        \right), \\[1ex]
    \mathbf{c}_{\text{new}} &= \mathbf{l}_1 \cdot \mathbf{l}_2, \\
    \mathbf{l}_5 &= \tanh\left( \mathbf{l}_3 + \mathbf{l}_4 \right), \\
    \mathbf{h}_{\text{new}} &= \tanh\left( \mathbf{c}_{\text{new}} \cdot
        \mathbf{l}_5 \right)
\end{aligned}
```

# Forward

    nas(inp, (state, cstate))
    nas(inp)

## Arguments
- `inp`: The input to the nas. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the NAS. 
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct NAS{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand NAS

function NAS((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = NASCell(input_size => hidden_size; kwargs...)
    return NAS{return_state, typeof(cell)}(cell)
end

function functor(rnn::NAS{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> NAS{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, nas::NAS)
    print(io, "NAS(", size(nas.cell.Wi, 2), " => ", size(nas.cell.Wi, 1))
    print(io, ")")
end
