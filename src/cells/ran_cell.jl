#https://arxiv.org/pdf/1705.07393
@doc raw"""
    RANCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Recurrent Additive Network cell [Lee2017](@cite).
See [`RAN`](@ref) for a layer that processes entire sequences.

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
    \tilde{\mathbf{c}}(t) &= \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{c}, \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{i}(t) \odot \tilde{\mathbf{c}}(t) +
        \mathbf{f}(t) \odot \mathbf{c}(t-1), \\
    \mathbf{h}(t) &= g\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    rancell(inp, (state, cstate))
    rancell(inp)

## Arguments
- `inp`: The input to the rancell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the RANCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct RANCell{I, H, V, W, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer RANCell

function RANCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(3 * hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(2 * hidden_size))
    else
        weight_hh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return RANCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (ran::RANCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(ran, inp, 1 => size(ran.weight_ih, 2))
    proj_ih = dense_proj(ran.weight_ih, inp, ran.bias_ih)
    proj_hh = dense_proj(ran.weight_hh, state, ran.bias_hh)
    gxs = chunk(proj_ih, 3; dims=1)
    ghs = chunk(proj_hh, 2; dims=1)
    input_gate = sigmoid_fast.(ran.integration_fn(gxs[1], ghs[1]))
    forget_gate = sigmoid_fast.(ran.integration_fn(gxs[2], ghs[2]))
    candidate_state = @. input_gate * gxs[3] + forget_gate * c_state
    new_state = @. tanh_fast(candidate_state)
    return new_state, (new_state, candidate_state)
end

function initialstates(ran::RANCell)
    state = zeros_like(ran.weight_hh, size(ran.weight_hh, 1) ÷ 2)
    second_state = zeros_like(ran.weight_hh, size(ran.weight_hh, 1) ÷ 2)
    return state, second_state
end

function Base.show(io::IO, ran::RANCell)
    print(io, "RANCell(", size(ran.weight_ih, 2), " => ", size(ran.weight_ih, 1) ÷ 3, ")")
end

@doc raw"""
    RAN(input_size => hidden_size;
        return_state = false, kwargs...)

Recurrent Additive Network cell [Lee2017](@cite).
See [`RANCell`](@ref) for a layer that processes a single sequence.

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
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations
```math
\begin{aligned}
    \tilde{\mathbf{c}}(t) &= \mathbf{W}^{c}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{c} \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}^{i}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{i}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{i} \right) \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}^{f}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{f} \right) \\
    \mathbf{c}(t) &= \mathbf{i}(t) \odot \tilde{\mathbf{c}}(t) +
        \mathbf{f}(t) \odot \mathbf{c}(t-1) \\
    \mathbf{h}(t) &= g\left( \mathbf{c}(t) \right)
\end{aligned}
```

# Forward

    ran(inp, (state, cstate))
    ran(inp)

## Arguments
- `inp`: The input to the ran. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the RAN.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct RAN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand RAN

function RAN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = RANCell(input_size => hidden_size; kwargs...)
    return RAN{return_state, typeof(cell)}(cell)
end

function functor(rnn::RAN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> RAN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, ran::RAN)
    print(io, "RAN(", size(ran.cell.weight_ih, 2), " => ", size(ran.cell.weight_ih, 1))
    print(io, ")")
end
