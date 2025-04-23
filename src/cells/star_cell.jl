#https://arxiv.org/abs/1911.11033
@doc raw"""
    STARCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Stackable recurrent cell](https://arxiv.org/abs/1911.11033).
See [`STAR`](@ref) for a layer that processes entire sequences.

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
    z_t &= \tanh(W_z x_t + b_z), \\
    k_t &= \sigma(W_x x_t + W_h h_{t-1} + b_k), \\
    h_t &= \tanh\left((1 - k_t) \circ h_{t-1} + k_t \circ z_t\right).
\end{aligned}
```

# Forward

    starcell(inp, state)
    starcell(inp)

## Arguments
- `inp`: The input to the rancell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the STARCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct STARCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

function STARCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return STARCell(Wi, Wh, b)
end

function (star::STARCell)(inp::AbstractVecOrMat, state)
    _size_check(star, inp, 1 => size(star.Wi, 2))
    Wi, Wh, b = star.Wi, star.Wh, star.bias
    #split
    gxs = chunk(Wi * inp .+ b, 2; dims=1)
    gh = Wh * state
    #compute
    input_gate = tanh_fast.(gxs[1])
    forget_gate = @. sigmoid_fast(gxs[2] + gh)
    new_state = @. tanh_fast((1 - forget_gate) * state + forget_gate * input_gate)

    return new_state, new_state
end

function Base.show(io::IO, star::STARCell)
    print(io, "STARCell(", size(star.Wi, 2), " => ", size(star.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    STAR(input_size => hidden_size;
        return_state = false, kwargs...)

[Stackable recurrent network](https://arxiv.org/abs/1911.11033).
See [`STARCell`](@ref) for a layer that processes a single sequence.

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
    z_t &= \tanh(W_z x_t + b_z), \\
    k_t &= \sigma(W_x x_t + W_h h_{t-1} + b_k), \\
    h_t &= \tanh\left((1 - k_t) \circ h_{t-1} + k_t \circ z_t\right).
\end{aligned}
```

# Forward

    star(inp, state)
    star(inp)

## Arguments
- `inp`: The input to the star. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the STAR. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct STAR{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand STAR

function STAR((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = STARCell(input_size => hidden_size; kwargs...)
    return STAR{return_state, typeof(cell)}(cell)
end

function functor(star::STAR{S}) where {S}
    params = (cell=star.cell,)
    reconstruct = p -> STAR{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, star::STAR)
    print(io, "STAR(", size(star.cell.Wi, 2), " => ", size(star.cell.Wi, 1))
    print(io, ")")
end
