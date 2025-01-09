#https://www.mdpi.com/2079-9292/13/16/3204

@doc raw"""
    LightRUCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Light recurrent unit](https://www.mdpi.com/2079-9292/13/16/3204).
See [`LightRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \tanh(W_h x_t), \\
f_t         &= \delta(W_f x_t + U_f h_{t-1} + b_f), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t.
\end{aligned}
```

# Forward

    lightrucell(inp, state)
    lightrucell(inp)

## Arguments
- `inp`: The input to the lightrucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the LightRUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct LightRUCell{I,H,V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer LightRUCell

function LightRUCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)
    Wi = init_kernel(2 * hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))

    return LightRUCell(Wi, Wh, b)
end

function (lightru::LightRUCell)(inp::AbstractVecOrMat, state)
    _size_check(lightru, inp, 1 => size(lightru.Wi,2))
    Wi, Wh, b = lightru.Wi, lightru.Wh, lightru.bias

    #split
    gxs = chunk(Wi * inp, 2, dims=1)

    #compute
    candidate_state = @. tanh_fast(gxs[1])
    forget_gate = sigmoid_fast(gxs[2] .+ Wh * state .+ b)
    new_state = @. (1 - forget_gate) * state + forget_gate * candidate_state
    return new_state, new_state
end

Base.show(io::IO, lightru::LightRUCell) =
    print(io, "LightRUCell(", size(lightru.Wi, 2), " => ", size(lightru.Wi, 1)÷2, ")")


@doc raw"""
    LightRU((input_size => hidden_size);
        return_state = false, kwargs...)

[Light recurrent unit network](https://www.mdpi.com/2079-9292/13/16/3204).
See [`LightRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `return_state`: Option to return the last state together with the output. Default is `false`.
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \tanh(W_h x_t), \\
f_t         &= \delta(W_f x_t + U_f h_{t-1} + b_f), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t.
\end{aligned}
```

# Forward

    lightru(inp, state)
    lightru(inp)

## Arguments
- `inp`: The input to the lightru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the LightRU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct LightRU{S,M} <: AbstractRecurrentLayer
    cell::M
end
  
@layer :noexpand LightRU

function LightRU((input_size, hidden_size)::Pair;
    return_state = false,
    kwargs...)
    cell = LightRUCell(input_size => hidden_size; kwargs...)
    return LightRU{return_state, typeof(cell)}(cell)
end

function Base.show(io::IO, lightru::LightRU)
    print(io, "LightRU(", size(lightru.cell.Wi, 2), " => ", size(lightru.cell.Wi, 1))
    print(io, ")")
end