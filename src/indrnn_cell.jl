#https://arxiv.org/pdf/1803.04831
struct IndRNNCell{F,I,H,V} <: AbstractRecurrentCell
    σ::F
    Wi::I
    Wh::H
    b::V
end

Flux.@layer IndRNNCell

@doc raw"""
    IndRNNCell((input_size => hidden_size)::Pair, σ=relu;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)


[Independently recurrent cell](https://arxiv.org/pdf/1803.04831).
See [`IndRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `σ`: activation function. Default is `tanh`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\mathbf{h}_{t} = \sigma(\mathbf{W} \mathbf{x}_t + \mathbf{u} \odot \mathbf{h}_{t-1} + \mathbf{b})
```

# Forward

    indrnncell(inp, state)
    indrnncell(inp)

## Arguments
- `inp`: The input to the indrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the IndRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
function IndRNNCell((input_size, hidden_size)::Pair, σ=relu;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return IndRNNCell(σ, Wi, Wh, b)
end

function (indrnn::IndRNNCell)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(indrnn, inp, 1 => size(indrnn.Wi, 2))
    σ = NNlib.fast_act(indrnn.σ, inp)
    state = σ.(indrnn.Wi*inp .+ indrnn.Wh .* state .+ indrnn.b)
    return state, state
end

function Base.show(io::IO, indrnn::IndRNNCell)
    print(io, "IndRNNCell(", size(indrnn.Wi, 2), " => ", size(indrnn.Wi, 1))
    print(io, ", ", indrnn.σ)
    print(io, ")")
end

struct IndRNN{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :expand IndRNN

@doc raw"""
    IndRNN((input_size, hidden_size)::Pair, σ = tanh, σ=relu;
        kwargs...)

[Independently recurrent network](https://arxiv.org/pdf/1803.04831).
See [`IndRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `σ`: activation function. Default is `tanh`
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\mathbf{h}_{t} = \sigma(\mathbf{W} \mathbf{x}_t + \mathbf{u} \odot \mathbf{h}_{t-1} + \mathbf{b})
```
# Forward

    indrnn(inp, state)
    indrnn(inp)

## Arguments
- `inp`: The input to the indrnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the IndRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function IndRNN((input_size, hidden_size)::Pair, σ = tanh; kwargs...)
    cell = IndRNNCell(input_size, hidden_size, σ; kwargs...)
    return IndRNN(cell)
end
  
function (indrnn::IndRNN)(inp, state) 
    @assert ndims(inp) == 2 || ndims(inp) == 3
    return scan(indrnn.cell, inp, state)
end