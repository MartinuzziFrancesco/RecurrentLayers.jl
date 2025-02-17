#https://arxiv.org/abs/1602.02218
@doc raw"""
    TRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Strongly typed recurrent unit](https://arxiv.org/abs/1602.02218).
See [`TRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: activation functio. Default is `tanh`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}

\end{aligned}
```

# Forward

    trnncell(inp, state)
    trnncell(inp)

## Arguments
- `inp`: The input to the trnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the TRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct TRNNCell{I, V, A} <: AbstractRecurrentCell
    Wi::I
    bias::V
    activation::A
end

@layer TRNNCell

function TRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, activation=tanh_fast;
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 2, input_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    return TRNNCell(Wi, b, activation)
end

function (trnn::TRNNCell)(inp::AbstractVecOrMat, state)
    _size_check(trnn, inp, 1 => size(trnn.Wi, 2))
    Wi, b, activation = trnn.Wi, trnn.bias, trnn.activation
    #split
    gxs = chunk(Wi * inp .+ b, 2; dims=1)

    forget_gate = activation.(gxs[2])
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* gxs[1]
    return new_state, new_state
end

function initialstates(trnn::TRNNCell)
  return zeros_like(trnn.Wi, size(trnn.Wi, 1) ÷ 2)
end

function Base.show(io::IO, trnn::TRNNCell)
    print(io, "TRNNCell(", size(trnn.Wi, 2), " => ", size(trnn.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    TRNN(input_size => hidden_size, [activation];
        return_state = false, kwargs...)

[Strongly typed recurrent unit](https://arxiv.org/abs/1602.02218).
See [`TRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: activation functio. Default is `tanh`.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}

\end{aligned}
```

# Forward

    trnn(inp, state)
    trnn(inp)

## Arguments
- `inp`: The input to the trnn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the TRNN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct TRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand TRNN

function TRNN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = TRNNCell(input_size => hidden_size; kwargs...)
    return TRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::TRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> TRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, trnn::TRNN)
    print(io, "TRNN(", size(trnn.cell.Wi, 2), " => ", size(trnn.cell.Wi, 1) ÷ 2)
    print(io, ")")
end
