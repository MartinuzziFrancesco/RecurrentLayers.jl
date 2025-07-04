#https://arxiv.org/pdf/1603.09420
@doc raw"""
    MGUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

Minimal gated unit [Zhou2016](@cite).
See [`MGU`](@ref) for a layer that processes entire sequences.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \left( \mathbf{f}(t) \odot \mathbf{h}(t-1) \right) +
        \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{f}(t)\right) \odot \mathbf{h}(t-1) +
        \mathbf{f}(t) \odot \tilde{\mathbf{h}}(t)
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
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct MGUCell{I, H, V} <: AbstractRecurrentCell
    Wi::I
    Wh::H
    bias::V
end

@layer MGUCell

function MGUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size * 2, input_size)
    Wh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))

    return MGUCell(Wi, Wh, b)
end

function (mgu::MGUCell)(inp::AbstractVecOrMat, state)
    _size_check(mgu, inp, 1 => size(mgu.Wi, 2))
    Wi, Wh, b = mgu.Wi, mgu.Wh, mgu.bias
    #split
    gxs = chunk(Wi * inp .+ b, 2; dims=1)
    ghs = chunk(Wh, 2; dims=1)
    t_ones = eltype(Wi)(1.0f0)

    forget_gate = sigmoid_fast.(gxs[1] .+ ghs[1] * state)
    candidate_state = tanh_fast.(gxs[2] .+ ghs[2] * (forget_gate .* state))
    new_state = @. forget_gate * state + (t_ones - forget_gate) * candidate_state
    return new_state, new_state
end

function Base.show(io::IO, mgu::MGUCell)
    print(io, "MGUCell(", size(mgu.Wi, 2), " => ", size(mgu.Wi, 1) ÷ 2, ")")
end

@doc raw"""
    MGU(input_size => hidden_size;
        return_state = false, kwargs...)

Minimal gated unit network [Zhou2016](@cite).
See [`MGUCell`](@ref) for a layer that processes a single sequence.

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
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{h}_{hh} \left( \mathbf{f}(t) \odot \mathbf{h}(t-1) \right) +
        \mathbf{b}^{h} \right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{f}(t)\right) \odot \mathbf{h}(t-1) +
        \mathbf{f}(t) \odot \tilde{\mathbf{h}}(t)
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
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct MGU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MGU

function MGU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MGUCell(input_size => hidden_size; kwargs...)
    return MGU{return_state, typeof(cell)}(cell)
end

function functor(rnn::MGU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MGU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, mgu::MGU)
    print(io, "MGU(", size(mgu.cell.Wi, 2), " => ", size(mgu.cell.Wi, 1))
    print(io, ")")
end
