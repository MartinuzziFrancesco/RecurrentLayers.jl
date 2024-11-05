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

struct NASCell{I,H,V}
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer NASCell

"""
    NASCell((in, out)::Pair;
        kernel_init = glorot_uniform,
        recurrent_kernel_init = glorot_uniform,
        bias = true)
"""
function NASCell((in, out)::Pair;
    kernel_init = glorot_uniform,
    recurrent_kernel_init = glorot_uniform,
    bias = true)
    Wi = kernel_init(8 * out, in)
    Wh = recurrent_kernel_init(8 * out, out)
    b = create_bias(Wi, bias, size(Wh, 1))
    return NASCell(Wi, Wh, b)
end

function (nas::NASCell)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(nas.Wh, 2))
    c_state = zeros_like(state)
    return nas(inp, (state, c_state))
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

    return new_state, new_cstate
end

Base.show(io::IO, nas::NASCell) =
    print(io, "NASCell(", size(nas.Wi, 2), " => ", size(nas.Wi, 1)÷8, ")")


struct NAS{M}
    cell::M
end

Flux.@layer :expand NAS

"""
    NAS((in, out)::Pair; kwargs...)
"""
function NAS((in, out)::Pair; kwargs...)
    cell = NASCell(in => out; kwargs...)
    return NAS(cell)
end

function (nas::NAS)(inp)
    state = zeros_like(inp, size(nas.cell.Wh, 2))
    c_state = zeros_like(state)
    return nas(inp, (state, c_state))
end

function (nas::NAS)(inp, (state, c_state))
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    new_cstate = []
    for inp_t in eachslice(inp, dims=2)
        state, c_state = nas.cell(inp_t, (state, c_state))
        new_state = vcat(new_state, [state])
        new_cstate = vcat(new_cstate, [c_state])
    end
    return stack(new_state, dims=2), stack(new_cstate, dims=2)
end
