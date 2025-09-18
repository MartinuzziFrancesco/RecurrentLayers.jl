#https://arxiv.org/abs/1804.04849
@doc raw"""
    JANETCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition,
        beta_value=1.0)

Just another network unit [Westhuizen2018](@cite).
See [`JANET`](@ref) for a layer that processes entire sequences.

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
- `beta_value`: control over the input data flow.
  Default is 1.0.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{f}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{f}, \\
    \tilde{\mathbf{c}}(t) &= \tanh\left( \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) +
        \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{s}(t) \right) \odot \mathbf{c}(t-1) +
        \left( 1 - \sigma\left( \mathbf{s}(t) - \beta \right) \right) \odot
        \tilde{\mathbf{c}}(t), \\
    \mathbf{h}(t) &= \mathbf{c}(t).
\end{aligned}
```

# Forward

    janetcell(inp, (state, cstate))
    janetcell(inp)

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
struct JANETCell{I, H, V, W, M, B} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::M
    beta::B
end

@layer JANETCell trainable=(weight_ih, weight_hh, bias_ih, bias_hh)

function JANETCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false, beta_value::AbstractFloat=1.0f0)
    weight_ih = init_kernel(hidden_size * 2, input_size)
    if independent_recurrence
        weight_hh = vec(init_recurrent_kernel(2 * hidden_size))
    else
        weight_hh = init_recurrent_kernel(hidden_size * 2, hidden_size)
    end
    beta = fill(eltype(weight_ih)(beta_value), hidden_size)
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
    return JANETCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn, beta)
end

function (janet::JANETCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(janet, inp, 1 => size(janet.weight_ih, 2))
    proj_ih = dense_proj(janet.weight_ih, inp, janet.bias_ih)
    proj_hh = dense_proj(janet.weight_hh, state, janet.bias_hh)
    gxs = chunk(proj_ih, 2; dims=1)
    ghs = chunk(proj_hh, 2; dims=1)
    linear_gate = janet.integration_fn(gxs[1], ghs[1])
    candidate_state = tanh_fast.(janet.integration_fn(gxs[2], ghs[2]))
    t_ones = eltype(candidate_state)(1.0f0)
    new_cstate = @. sigmoid_fast(linear_gate) * c_state +
                    (t_ones -
                     sigmoid_fast(linear_gate - janet.beta)) *
                    candidate_state
    new_state = new_cstate

    return new_state, (new_state, new_cstate)
end

function initialstates(janet::JANETCell)
    state = zeros_like(janet.weight_hh, size(janet.weight_hh, 1) ÷ 2)
    second_state = zeros_like(janet.weight_hh, size(janet.weight_hh, 1) ÷ 2)
    return state, second_state
end

function Base.show(io::IO, janet::JANETCell)
    print(io, "JANETCell(", size(janet.weight_ih, 2),
        " => ", size(janet.weight_ih, 1) ÷ 2, ")")
end

@doc raw"""
    JANET(input_size => hidden_size;
        return_state = false, kwargs...)

Just another network [Westhuizen2018](@cite).
See [`JANETCell`](@ref) for a layer that processes a single sequence.

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
- `beta_value`: control over the input data flow.
  Default is 1.0.

# Equations

```math
\begin{aligned}
    \mathbf{s}(t) &= \mathbf{W}^{f}_{hh} \mathbf{h}(t-1) + \mathbf{W}^{f}_{ih}
        \mathbf{x}(t) + \mathbf{b}^{f}, \\
    \tilde{\mathbf{c}}(t) &= \tanh\left( \mathbf{W}^{c}_{hh} \mathbf{h}(t-1) +
        \mathbf{W}^{c}_{ih} \mathbf{x}(t) + \mathbf{b}^{c} \right), \\
    \mathbf{c}(t) &= \sigma\left( \mathbf{s}(t) \right) \odot \mathbf{c}(t-1) +
        \left( 1 - \sigma\left( \mathbf{s}(t) - \beta \right) \right) \odot
        \tilde{\mathbf{c}}(t), \\
    \mathbf{h}(t) &= \mathbf{c}(t).
\end{aligned}
```

# Forward

    janet(inp, (state, cstate))
    janet(inp)

## Arguments
- `inp`: The input to the janet. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the JANET.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct JANET{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand JANET

function JANET((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = JANETCell(input_size => hidden_size; kwargs...)
    return JANET{return_state, typeof(cell)}(cell)
end

function functor(janet::JANET{S}) where {S}
    params = (cell=janet.cell,)
    reconstruct = p -> JANET{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, janet::JANET)
    print(
        io, "JANET(", size(janet.cell.weight_ih, 2), " => ", size(janet.cell.weight_ih, 1))
    print(io, ")")
end
