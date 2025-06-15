#https://arxiv.org/abs/1612.06212
@doc raw"""
    CFNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Chaos free network unit](https://arxiv.org/abs/1612.06212).
See [`CFN`](@ref) for a layer that processes entire sequences.

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
    \mathbf{h}(t) &= \boldsymbol{\theta}(t) \odot \tanh\left( \mathbf{h}(t-1)
        \right) + \boldsymbol{\eta}(t) \odot \tanh\left( \mathbf{W}_{ih}
        \mathbf{x}(t) \right), \\
    \boldsymbol{\theta}(t) &= \sigma\left( \mathbf{W}^{\theta}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}^{\theta}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{\theta} \right), \\
    \boldsymbol{\eta}(t) &= \sigma\left( \mathbf{W}^{\eta}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}^{\eta}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{\eta} \right).
\end{aligned}
```

# Forward

    cfncell(inp, state)
    cfncell(inp)

## Arguments
- `inp`: The input to the cfncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the CFNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct CFNCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer CFNCell

function CFNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 3, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))

    return CFNCell(Wi, Wh, b)
end

function (cfn::CFNCell)(inp::AbstractVecOrMat, state)
    _size_check(cfn, inp, 1 => size(cfn.Wi, 2))
    Wi, Wh, b = cfn.Wi, cfn.Wh, cfn.bias
    #split
    gxs = chunk(Wi * inp, 3; dims=1)
    ghs = chunk(Wh * state .+ b, 2; dims=1)

    horizontal_gate = @. sigmoid_fast(gxs[1] + ghs[1])
    vertical_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    new_state = @. horizontal_gate * tanh_fast(state) + vertical_gate * tanh_fast(gxs[3])
    return new_state, new_state
end

function Base.show(io::IO, cfn::CFNCell)
    print(io, "CFNCell(", size(cfn.Wi, 2), " => ", size(cfn.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    CFN(input_size => hidden_size;
        return_state = false, kwargs...)

[Chaos free network unit](https://arxiv.org/abs/1612.06212).
See [`CFNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations

```math
\begin{aligned}
    \mathbf{h}(t) &= \boldsymbol{\theta}(t) \odot \tanh\left( \mathbf{h}(t-1)
        \right) + \boldsymbol{\eta}(t) \odot \tanh\left( \mathbf{W}_{ih}
        \mathbf{x}(t) \right), \\
    \boldsymbol{\theta}(t) &= \sigma\left( \mathbf{W}^{\theta}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}^{\theta}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{\theta} \right), \\
    \boldsymbol{\eta}(t) &= \sigma\left( \mathbf{W}^{\eta}_{hh}
        \mathbf{h}(t-1) + \mathbf{W}^{\eta}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{\eta} \right).
\end{aligned}
```

# Forward

    cfn(inp, state)
    cfn(inp)

## Arguments
- `inp`: The input to the cfn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the CFN. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct CFN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand CFN

function CFN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = CFNCell(input_size => hidden_size; kwargs...)
    return CFN{return_state, typeof(cell)}(cell)
end

function functor(rnn::CFN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> CFN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, cfn::CFN)
    print(io, "CFN(", size(cfn.cell.Wi, 2), " => ", size(cfn.cell.Wi, 1))
    print(io, ")")
end
