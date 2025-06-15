#https://arxiv.org/abs/2103.05487
@doc raw"""
    UnICORNNCell(input_size => hidden_size, [dt];
        alpha=0.0, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform, bias = true)

[Undamped independent controlled oscillatory recurrent neural unit](https://arxiv.org/abs/2103.05487).
See [`coRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `alpha`: Control parameter. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

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
struct UnICORNNCell{I, H, Z, V, D, A} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    c::Z
    bias::V
    dt::D
    alpha::A
end

@layer UnICORNNCell

function UnICORNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        dt::Number=1.0f0; alpha::Number=0.0f0,
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true)
    Wi = init_kernel(hidden_size, input_size)
    Wh = init_recurrent_kernel(hidden_size)
    c = init_kernel(hidden_size)
    b = create_bias(Wi, bias, size(Wi, 1))
    T = eltype(Wi)
    return UnICORNNCell(Wi, Wh, c, b, T(dt), T(alpha))
end

function (unicornn::UnICORNNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(unicornn, inp, 1 => size(unicornn.Wi, 2))
    Wi, Wh, c, b = unicornn.Wi, unicornn.Wh, unicornn.c, unicornn.bias
    dt, alpha = unicornn.dt, unicornn.alpha
    new_cstate = c_state .-
                 dt .* sigmoid_fast.(c) .*
                 (tanh_fast.(Wh .* state .+ Wi * inp .+ b) .+ alpha .* state)
    new_state = state .+ dt .* sigmoid_fast.(c) .* new_cstate
    return new_state, (new_state, new_cstate)
end

function initialstates(unicornn::UnICORNNCell)
    state = zeros_like(unicornn.Wi, size(unicornn.Wi, 1))
    c_state = zeros_like(unicornn.Wi, size(unicornn.Wi, 1))
    return state, c_state
end

function Base.show(io::IO, unicornn::UnICORNNCell)
    print(io, "UnICORNNCell(", size(unicornn.Wi, 2), " => ", size(unicornn.Wi, 1), ")")
end

@doc raw"""
    UnICORNN(input_size => hidden_size, [dt];
        alpha=0.0, return_state=false, init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform, bias = true)

[Undamped independent controlled oscillatory recurrent neural network](https://arxiv.org/abs/2010.00951).
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
- `bias`: include a bias or not. Default is `true`.
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
    print(io, "UnICORNN(", size(unicornn.cell.Wi, 2),
        " => ", size(unicornn.cell.Wi, 1))
    print(io, ")")
end
