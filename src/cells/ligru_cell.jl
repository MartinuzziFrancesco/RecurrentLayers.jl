#https://arxiv.org/pdf/1803.10225
@doc raw"""
    LiGRUCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Light gated recurrent unit [Ravanelli2018](@cite).
The implementation does not include the batch normalization as
described in the original paper.
See [`LiGRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \tilde{\mathbf{h}}(t) &= \text{ReLU}\left( \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{W}^{h}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{h}
        \right), \\
    \mathbf{h}(t) &= \mathbf{z}(t) \odot \mathbf{h}(t-1) + \left(1 -
        \mathbf{z}(t)\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    ligrucell(inp, state)
    ligrucell(inp)

## Arguments
- `inp`: The input to the ligrucell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the LiGRUCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
struct LiGRUCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer LiGRUCell

function LiGRUCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return LiGRUCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (ligru::LiGRUCell)(inp::AbstractVecOrMat, state)
    _size_check(ligru, inp, 1 => size(ligru.weight_ih, 2))
    proj_ih = dense_proj(ligru.weight_ih, inp, ligru.bias_ih)
    proj_hh = dense_proj(ligru.weight_hh, state, ligru.bias_hh)
    gxs = chunk(proj_ih, 2; dims=1)
    ghs = chunk(proj_hh, 2; dims=1)
    forget_gate = sigmoid_fast.(ligru.integration_fn(gxs[1], ghs[1]))
    candidate_hidden = tanh_fast.(ligru.integration_fn(gxs[2], ghs[2]))
    new_state = @. forget_gate * state + (1 - forget_gate) * candidate_hidden
    return new_state, new_state
end

function initialstates(ligru::LiGRUCell)
    return zeros_like(ligru.weight_hh, size(ligru.weight_hh, 1) ÷ 2)
end

function Base.show(io::IO, ligru::LiGRUCell)
    print(io, "LiGRUCell(", size(ligru.weight_ih, 2),
        " => ", size(ligru.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    LiGRU(input_size => hidden_size;
        return_state = false, kwargs...)

Light gated recurrent network [Ravanelli2018](@cite).
The implementation does not include the batch normalization as
described in the original paper.
See [`LiGRUCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `return_state`: Option to return the last state together with the output.
  Default is `false`.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{z}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{z} \right), \\
    \tilde{\mathbf{h}}(t) &= \text{ReLU}\left( \mathbf{W}^{h}_{ih}
        \mathbf{x}(t) + \mathbf{W}^{h}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{h}
        \right), \\
    \mathbf{h}(t) &= \mathbf{z}(t) \odot \mathbf{h}(t-1) + \left(1 -
        \mathbf{z}(t)\right) \odot \tilde{\mathbf{h}}(t)
\end{aligned}
```

# Forward

    ligru(inp, state)
    ligru(inp)

## Arguments
- `inp`: The input to the ligru. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `state`: The hidden state of the LiGRU. If given, it is a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct LiGRU{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand LiGRU

function LiGRU((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = LiGRUCell(input_size => hidden_size; kwargs...)
    return LiGRU{return_state, typeof(cell)}(cell)
end

function functor(rnn::LiGRU{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> LiGRU{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, ligru::LiGRU)
    print(
        io, "LiGRU(", size(ligru.cell.weight_ih, 2), " => ", size(ligru.cell.weight_ih, 1))
    print(io, ")")
end
