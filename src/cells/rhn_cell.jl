#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60

struct RHNCellUnit{I, V}
    weights::I
    bias::V
    num_gates::Int
end

@layer RHNCellUnit

function RHNCellUnit((input_size, hidden_size)::Pair{<:Int, <:Int}, num_gates::Int=3;
        init_kernel=glorot_uniform, bias::Bool=true)
    weight = init_kernel(num_gates * hidden_size, input_size)
    b = create_bias(weight, bias, size(weight, 1))
    return RHNCellUnit(weight, b, num_gates)
end

function initialstates(rhn::RHNCellUnit)
    return zeros_like(rhn.weights, size(rhn.weights, 1) ÷ rhn.num_gates)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat)
    state = initialstates(rhn)
    return rhn(inp, state)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    _size_check(rhn, inp, 1 => size(rhn.weights, 2))
    weight, bias = rhn.weights, rhn.bias
    #compute
    pre_nonlin = weight * inp .+ bias
    #split
    return chunk(pre_nonlin, rhn.num_gates; dims=1)
end

function Base.show(io::IO, rhn::RHNCellUnit)
    print(io, "RHNCellUnit(", size(rhn.weights, 2), " => ",
        size(rhn.weights, 1) ÷ rhn.num_gates, ")")
end

@doc raw"""
    RHNCell(input_size => hidden_size, [depth];
        couple_carry = true,
        cell_kwargs...)

Recurrent highway network [Zilly2017](@cite).
See [`RHN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `depth`: depth of the recurrence. Default is 3.

# Keyword arguments

- `couple_carry`: couples the carry gate and the transform gate. Default `true`
- `init_kernel`: initializer for the input to hidden weights.
  Default is `glorot_uniform`
- `bias`: include a bias or not. Default is `true`

# Equations

```math
\begin{aligned}
    \mathbf{s}_{\ell}(t) &= \mathbf{h}_{\ell}(t) \odot \mathbf{t}_{\ell}(t) +
        \mathbf{s}_{\ell-1}(t) \odot \mathbf{c}_{\ell}(t) \\
    \mathbf{h}_{\ell}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{h_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{h_{\ell}} \right) \\
    \mathbf{t}_{\ell}(t) &= \sigma\left( \mathbf{W}^{t}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{t_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{t_{\ell}} \right) \\
    \mathbf{c}_{\ell}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{c_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{c_{\ell}} \right)
\end{aligned}
```

# Forward

    rnncell(inp, [state])
"""
struct RHNCell{C}
    layers::C
    couple_carry::Bool
end

@layer RHNCell

function RHNCell((input_size, hidden_size)::Pair{<:Int, <:Int}, depth::Integer=3;
        couple_carry::Bool=true, #sec 5, setup
        cell_kwargs...)
    depth > 0 || throw(ArgumentError("depth must be a positive integer; got $depth"))
    num_gates = couple_carry ? 2 : 3
    layers = ntuple(depth) do layer
        real_in = layer == 1 ? input_size + hidden_size : hidden_size
        RHNCellUnit(real_in => hidden_size, num_gates; cell_kwargs...)
    end
    return RHNCell(Chain(layers...), couple_carry)
end

function initialstates(rhn::RHNCell)
    return initialstates(first(rhn.layers))
end

function (rhn::RHNCell)(inp::AbstractArray)
    state = initialstates(rhn)
    return rhn(inp, state)
end

function (rhn::RHNCell)(inp::AbstractArray, state::AbstractVecOrMat)
    current_state = _rhn_batch_state(state, inp)
    layers = rhn.layers.layers

    # the first micro-layer has a differently-shaped input (x(t) is
    # concatenated in), so it is kept out of the loop below: mixing
    # differently-shaped iterations in one Julia `for` loop breaks Zygote's
    # reverse-mode AD (it tries to accumulate gradients of mismatched shapes).
    current_state = _rhn_layer_step(
        first(layers), vcat(inp, current_state), current_state, rhn.couple_carry)
    for layer in Base.tail(layers)
        current_state = _rhn_layer_step(layer, current_state, current_state, rhn.couple_carry)
    end

    return current_state, current_state
end

function _rhn_layer_step(layer, inp_combined, current_state, couple_carry::Bool)
    if couple_carry
        pre_h, pre_t = layer(inp_combined)
        hidden_gate = tanh_fast.(pre_h)
        transform_gate = sigmoid_fast.(pre_t)
        return @. (hidden_gate - current_state) * transform_gate + current_state
    else
        pre_h, pre_t, pre_c = layer(inp_combined)
        hidden_gate = tanh_fast.(pre_h)
        transform_gate = sigmoid_fast.(pre_t)
        carry_gate = sigmoid_fast.(pre_c)
        return @. hidden_gate * transform_gate + current_state * carry_gate
    end
end

function _rhn_batch_state(state::AbstractVector, inp::AbstractMatrix)
    repeat(state, 1, size(inp, 2))
end
_rhn_batch_state(state::AbstractVecOrMat, inp::AbstractVector) = state
_rhn_batch_state(state::AbstractMatrix, inp::AbstractMatrix) = state

@doc raw"""
    RHN(input_size => hidden_size, [depth];
        return_state = false,
        kwargs...)

Recurrent highway network [Zilly2017](@cite).
See [`RHNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `depth`: depth of the recurrence. Default is 3

# Keyword arguments

- `couple_carry`: couples the carry gate and the transform gate. Default `true`
- `init_kernel`: initializer for the input to hidden weights
- `bias`: include a bias or not. Default is `true`
- `return_state`: Option to return the last state together with the output.
  Default is `false`.

# Equations

```math
\begin{aligned}
    \mathbf{s}_{\ell}(t) &= \mathbf{h}_{\ell}(t) \odot \mathbf{t}_{\ell}(t) +
        \mathbf{s}_{\ell-1}(t) \odot \mathbf{c}_{\ell}(t) \\
    \mathbf{h}_{\ell}(t) &= \tanh\left( \mathbf{W}^{h}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{h_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{h_{\ell}} \right) \\
    \mathbf{t}_{\ell}(t) &= \sigma\left( \mathbf{W}^{t}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{t_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{t_{\ell}} \right) \\
    \mathbf{c}_{\ell}(t) &= \sigma\left( \mathbf{W}^{c}_{ih} \mathbf{x}(t) \,
        \mathbb{I}_{\ell = 1} + \mathbf{W}^{c_{\ell}}_{hh} \mathbf{s}_{\ell-1}(t)
        + \mathbf{b}^{c_{\ell}} \right)
\end{aligned}
```
"""
struct RHN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand RHN

function RHN((input_size, hidden_size)::Pair{<:Int, <:Int}, depth::Integer=3;
        return_state::Bool=false, kwargs...)
    cell = RHNCell(input_size => hidden_size, depth; kwargs...)
    return RHN{return_state, typeof(cell)}(cell)
end

function Base.show(io::IO, rhn::RHN)
    unit = first(rhn.cell.layers.layers)
    hidden_size = size(unit.weights, 1) ÷ unit.num_gates
    input_size = size(unit.weights, 2) - hidden_size
    print(io, "RHN(", input_size, " => ", hidden_size)
    depth = length(rhn.cell.layers.layers)
    if depth != 3
        print(io, ", ", depth)
    end
    print(io, ")")
end
