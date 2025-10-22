#https://arxiv.org/pdf/1412.7753

@doc raw"""
    SCRNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_context_kernel = glorot_uniform,
        bias = true, recurrent_bias = true, context_bias=true,
        independent_recurrence = false, integration_mode = :addition, alpha = 0.0)

Structurally contraint recurrent unit [Mikolov2014](@cite).
See [`SCRN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_context_kernel`: initializer for the context to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `context_bias`: include context to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `alpha`: structural contraint. Default is 0.0.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= (1 - \alpha) \, \mathbf{W}_{ih}^{s} \mathbf{x}(t) +
        \alpha \, \mathbf{s}(t-1) \\
    \mathbf{h}(t) &= \sigma\left( \mathbf{W}_{ih}^{h} \mathbf{s}(t) +
        \mathbf{W}_{hh}^{h} \mathbf{h}(t-1) + \mathbf{b}^{h} \right) \\
    \mathbf{y}(t) &= f\left( \mathbf{W}_{hh}^{y} \mathbf{h}(t) +
        \mathbf{W}_{ih}^{y} \mathbf{s}(t) + \mathbf{b}^{y} \right)
\end{aligned}
```

# Forward

    scrncell(inp, (state, cstate))
    scrncell(inp)

## Arguments

- `inp`: The input to the scrncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the SCRNCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct SCRNCell{I, H, C, V, W, K, A, O} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ch::C
    bias_ih::V
    bias_hh::W
    bias_ch::K
    integration_fn::A
    alpha::O
end

@layer SCRNCell

function SCRNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_context_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true, context_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false, alpha=0.0f0)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 2)
    weight_ch = init_context_kernel(2 * hidden_size, hidden_size)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    bias_ch = create_bias(weight_ch, context_bias, size(weight_ch, 1))
    integration_fn = _integration_fn(integration_mode)
    return SCRNCell(weight_ih, weight_hh, weight_ch, bias_ih, bias_hh,
        bias_ch, integration_fn, [eltype(weight_ih)(alpha)])
end

function (scrn::SCRNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(scrn, inp, 1 => size(scrn.weight_ih, 2))
    proj_ih = dense_proj(scrn.weight_ih, inp, scrn.bias_ih)
    gxs = chunk(proj_ih, 2; dims=1)
    whs = chunk(scrn.weight_hh, 2; dims=1)
    bhs = chunk(scrn.bias_hh, 2; dims=1)
    #compute
    t_ones = eltype(scrn.weight_ih)(1.0f0)
    new_cstate = @. (t_ones - scrn.alpha) * gxs[1] + scrn.alpha * c_state
    proj_ch = dense_proj(scrn.weight_ch, new_cstate, scrn.bias_ch)
    gcs = chunk(proj_ch, 2; dims=1)
    proj_hh_1 = dense_proj(whs[1], state, bhs[1])
    hidden_layer = sigmoid_fast.(scrn.integration_fn(gxs[2], proj_hh_1) .+ gcs[1])
    proj_hh_2 = dense_proj(whs[2], hidden_layer, bhs[2])
    new_state = tanh_fast.(proj_hh_2 .+ gcs[2])
    return new_state, (new_state, new_cstate)
end

function initialstates(scrn::SCRNCell)
    state = zeros_like(scrn.weight_hh, size(scrn.weight_hh, 1) ÷ 2)
    second_state = zeros_like(scrn.weight_hh, size(scrn.weight_hh, 1) ÷ 2)
    return state, second_state
end

function Base.show(io::IO, scrn::SCRNCell)
    print(
        io, "SCRNCell(", size(scrn.weight_ih, 2), " => ", size(scrn.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    SCRN(input_size => hidden_size;
        return_state = false,
        kwargs...)

Structurally contraint recurrent unit [Mikolov2014](@cite).
See [`SCRNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_context_kernel`: initializer for the context to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include input to recurrent bias or not. Default is `true`.
- `recurrent_bias`: include recurrent to recurrent bias or not. Default is `true`.
- `context_bias`: include context to recurrent bias or not. Default is `true`.
- `independent_recurrence`: flag to toggle independent recurrence. If `true`, the
  recurrent to recurrent weights are a vector instead of a matrix. Default `false`.
- `integration_mode`: determines how the input and hidden projections are combined. The
  options are `:addition` and `:multiplicative_integration`. Defaults to `:addition`.
- `alpha`: structural contraint. Default is 0.0.
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= (1 - \alpha) \, \mathbf{W}_{ih}^{s} \mathbf{x}(t) +
        \alpha \, \mathbf{s}(t-1) \\
    \mathbf{h}(t) &= \sigma\left( \mathbf{W}_{ih}^{h} \mathbf{s}(t) +
        \mathbf{W}_{hh}^{h} \mathbf{h}(t-1) + \mathbf{b}^{h} \right) \\
    \mathbf{y}(t) &= f\left( \mathbf{W}_{hh}^{y} \mathbf{h}(t) +
        \mathbf{W}_{ih}^{y} \mathbf{s}(t) + \mathbf{b}^{y} \right)
\end{aligned}
```

# Forward

    scrn(inp, (state, cstate))
    scrn(inp)

## Arguments
- `inp`: The input to the scrn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the SCRN.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct SCRN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand SCRN

function SCRN((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = SCRNCell(input_size => hidden_size; kwargs...)
    return SCRN{return_state, typeof(cell)}(cell)
end

function functor(rnn::SCRN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> SCRN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, scrn::SCRN)
    print(
        io, "SCRN(", size(scrn.cell.weight_ih, 2), " => ", size(scrn.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end
