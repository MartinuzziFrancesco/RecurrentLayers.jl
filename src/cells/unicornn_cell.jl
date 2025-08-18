#https://arxiv.org/abs/2103.05487
@doc raw"""
    UnICORNNCell(input_size => hidden_size, [dt];
        alpha=0.0, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        init_control_kernel = glorot_uniform,
        bias = true, recurrent_bias = true,
        independent_recurrence = false, integration_mode = :addition)

Undamped independent controlled oscillatory recurrent neural
unit [Rusch2021b](@cite).
See [`UnICORNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `alpha`: Control parameter. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_control_kernel`: initializer for the control to hidden weights.
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
    \mathbf{z}(t) &= \mathbf{z}(t-1) - \Delta t \, \hat{\sigma}(\mathbf{c}) \odot \left[
        \sigma\left( \mathbf{w} \odot \mathbf{h}(t-1) +
        \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b} \right) +
        \alpha \, \mathbf{h}(t-1) \right] \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t \, \hat{\sigma}(\mathbf{c}) \odot
        \mathbf{z}(t)
\end{aligned}
```

# Forward

    unicornncell(inp, (state, cstate))
    unicornncell(inp)

## Arguments
- `inp`: The input to the unicornncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the UnICORNNCell.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct UnICORNNCell{I, H, Z, V, W, F, D, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    weight_ch::Z
    bias_ih::V
    bias_hh::W
    integration_fn::F
    dt::D
    alpha::A
end

@layer UnICORNNCell

function UnICORNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        dt::Number=1.0f0; alpha::Number=0.0f0,
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        init_control_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(hidden_size, input_size)
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
    weight_ch = vec(init_control_kernel(hidden_size))
    T = eltype(weight_ih)
    return UnICORNNCell(
        weight_ih, weight_hh, weight_ch, bias_ih, bias_hh, integration_fn, T(dt), T(alpha))
end

function (unicornn::UnICORNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(unicornn, inp, 1 => size(unicornn.weight_ih, 2))
    proj_ih = dense_proj(unicornn.weight_ih, inp, unicornn.bias_ih)
    proj_hh = dense_proj(unicornn.weight_hh, state, unicornn.bias_hh)
    merged_proj = unicornn.integration_fn(proj_ih, proj_hh)
    candidate_state = tanh_fast.(merged_proj) .+ unicornn.alpha .* state
    new_cstate = c_state .-
                 unicornn.dt .* sigmoid_fast.(unicornn.weight_ch) .* candidate_state
    new_state = state .+ unicornn.dt .* sigmoid_fast.(unicornn.weight_ch) .* new_cstate
    return new_state, (new_state, new_cstate)
end

function initialstates(unicornn::UnICORNNCell)
    state = zeros_like(unicornn.weight_ih, size(unicornn.weight_ih, 1))
    c_state = zeros_like(unicornn.weight_ih, size(unicornn.weight_ih, 1))
    return state, c_state
end

function Base.show(io::IO, unicornn::UnICORNNCell)
    print(io, "UnICORNNCell(", size(unicornn.weight_ih, 2),
        " => ", size(unicornn.weight_ih, 1), ")")
end

@doc raw"""
    UnICORNN(input_size => hidden_size, [dt];
        alpha=0.0, return_state=false, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform, bias = true)

Undamped independent controlled oscillatory recurrent neural
network [Rusch2021b](@cite).
See [`UnICORNNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `alpha`: Control parameter. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `init_control_kernel`: initializer for the control to hidden weights.
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
    \mathbf{z}(t) &= \mathbf{z}(t-1) - \Delta t \, \hat{\sigma}(\mathbf{c}) \odot \left[
        \sigma\left( \mathbf{w} \odot \mathbf{h}(t-1) +
        \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b} \right) +
        \alpha \, \mathbf{h}(t-1) \right] \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t \, \hat{\sigma}(\mathbf{c}) \odot
        \mathbf{z}(t)
\end{aligned}
```

# Forward

    unicornn(inp, (state, zstate))
    unicornn(inp)

## Arguments
- `inp`: The input to the `unicornn`. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the `UnICORNN`.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.
"""
struct UnICORNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand UnICORNN

function UnICORNN((input_size, hidden_size)::Pair{<:Int, <:Int}, args...;
        return_state::Bool=false, kwargs...)
    cell = UnICORNNCell(input_size => hidden_size, args...; kwargs...)
    return UnICORNN{return_state, typeof(cell)}(cell)
end

function functor(unicornn::UnICORNN{S}) where {S}
    params = (cell=unicornn.cell,)
    reconstruct = p -> UnICORNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, unicornn::UnICORNN)
    print(io, "UnICORNN(", size(unicornn.cell.weight_ih, 2),
        " => ", size(unicornn.cell.weight_ih, 1))
    print(io, ")")
end
