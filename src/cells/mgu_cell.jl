#https://arxiv.org/pdf/1603.09420
@doc raw"""
    MGUCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Minimal gated unit](https://arxiv.org/pdf/1603.09420).
See [`MGU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (f_t \odot h_{t-1}) + b_h), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t
\end{aligned}
```

# Forward

    mgucell(inp, state)
    mgucell(inp)

## Arguments
- `inp`: The input to the mgucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the MGUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state `new_state`, 
  a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MGUCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

Flux.@layer MGUCell

function MGUCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true)

    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return MGUCell(Wi, Wh, b)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat, state)
    _size_check(mgu, inp, 1 => size(mgu.Wi,2))
    Wi, Wh, b = mgu.Wi, mgu.Wh, mgu.bias
    #split
    gxs = chunk(Wi * inp .+ b, 2, dims=1)
    ghs = chunk(Wh, 2, dims=1)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1]*state)
    candidate_state = tanh_fast.(gxs[2] .+ ghs[2]*(forget_gate.*state))
    new_state = forget_gate .* state .+ (1 .- forget_gate) .* candidate_state
    return new_state, new_state
end

Base.show(io::IO, mgu::MGUCell) =
    print(io, "MGUCell(", size(mgu.Wi, 2), " => ", size(mgu.Wi, 1) ÷ 2, ")")


@doc raw"""
    MGU((input_size => hidden_size)::Pair; kwargs...)

[Minimal gated unit network](https://arxiv.org/pdf/1603.09420).
See [`MGUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (f_t \odot h_{t-1}) + b_h), \\
h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t
\end{aligned}
```

# Forward

    mgu(inp, state)
    mgu(inp)

## Arguments
- `inp`: The input to the mgu. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the MGU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
struct MGU{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand MGU

function MGU((input_size, hidden_size)::Pair; kwargs...)
    cell = MGUCell(input_size => hidden_size; kwargs...)
    return MGU(cell)
end

function Base.show(io::IO, mgu::MGU)
    print(io, "MGU(", size(mgu.cell.Wi, 2), " => ", size(mgu.cell.Wi, 1))
    print(io, ")")
end