#https://arxiv.org/abs/1911.11033
@doc raw"""
    STARCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Stackable recurrent cell [Turkoglu2021](@cite).
See [`STAR`](@ref) for a layer that processes entire sequences.

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
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{z} \right) \\
    \mathbf{k}(t) &= \sigma\left( \mathbf{W}^{k}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{k}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{k} \right) \\
    \mathbf{h}(t) &= \tanh\left( \left(1 - \mathbf{k}(t)\right) \circ
        \mathbf{h}(t-1) + \mathbf{k}(t) \circ \mathbf{z}(t) \right)
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
struct STARCell{I, H, V, W, A} <: AbstractRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

function STARCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(2 * hidden_size, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(hidden_size))
    else
        weight_hh = init_recurrent_kernel(hidden_size, hidden_size)
    end
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    if integration_mode == :addition
        integration_fn = add_projections
    elseif integration_mode == :multiplicative_integration
        integration_fn = mul_projections
    else
        throw(ArgumentError(
            "integration_mode must be :addition or :multiplicative_integration; got $integration_mode"
        ))
    end
    return STARCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (star::STARCell)(inp::AbstractVecOrMat, state)
    _size_check(star, inp, 1 => size(star.weight_ih, 2))
    proj_ih = dense_proj(star.weight_ih, inp, star.bias_ih)
    proj_hh = dense_proj(star.weight_hh, state, star.bias_hh)
    gxs = chunk(proj_ih, 2; dims=1)
    #compute
    input_gate = tanh_fast.(gxs[1])
    forget_gate = sigmoid_fast.(star.integration_fn(gxs[2], proj_hh))
    new_state = @. tanh_fast((1 - forget_gate) * state + forget_gate * input_gate)
    return new_state, new_state
end

function initialstates(star::STARCell)
    return zeros_like(star.weight_hh, size(star.weight_hh, 1))
end

function Base.show(io::IO, star::STARCell)
    print(
        io, "STARCell(", size(star.weight_ih, 2), " => ", size(star.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    STAR(input_size => hidden_size;
        return_state = false, kwargs...)

Stackable recurrent network [Turkoglu2021](@cite).
See [`STARCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
-- `bias`: include input to recurrent bias or not. Default is `true`.
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
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}^{z}_{ih} \mathbf{x}(t) +
        \mathbf{b}^{z} \right) \\
    \mathbf{k}(t) &= \sigma\left( \mathbf{W}^{k}_{ih} \mathbf{x}(t) +
        \mathbf{W}^{k}_{hh} \mathbf{h}(t-1) + \mathbf{b}^{k} \right) \\
    \mathbf{h}(t) &= \tanh\left( \left(1 - \mathbf{k}(t)\right) \circ
        \mathbf{h}(t-1) + \mathbf{k}(t) \circ \mathbf{z}(t) \right)
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
    print(
        io, "STAR(", size(star.cell.weight_ih, 2), " => ", size(star.cell.weight_ih, 1) ÷ 2)
    print(io, ")")
end
