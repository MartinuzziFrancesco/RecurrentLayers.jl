#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60

"""
    RHNCellUnit(input_size => hidden_size;
        init_kernel = glorot_uniform,
        bias = true)
"""
struct RHNCellUnit{I, V}
    weights::I
    bias::V
end

@layer RHNCellUnit

function RHNCellUnit((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, bias::Bool=true)
    weight = init_kernel(3 * hidden_size, input_size)
    b = create_bias(weight, bias, size(weight, 1))
    return RHNCellUnit(weight, b)
end

function initialstates(rhn::RHNCellUnit)
    return zeros_like(rhn.weights, size(rhn.weights, 1) ÷ 3)
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
    pre_h, pre_t, pre_c = chunk(pre_nonlin, 3; dims=1)
    return pre_h, pre_t, pre_c
end

function Base.show(io::IO, rhn::RHNCellUnit)
    print(io, "RHNCellUnit(", size(rhn.weights, 2), " => ", size(rhn.weights, 1) ÷ 3, ")")
end

@doc raw"""
    RHNCell(input_size => hidden_size, [depth];
        couple_carry = true,
        cell_kwargs...)

[Recurrent highway network](https://arxiv.org/pdf/1607.03474).
See [`RHNCellUnit`](@ref) for a the unit component of this layer.
See [`RHN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `depth`: depth of the recurrence. Default is 3

# Keyword arguments

- `couple_carry`: couples the carry gate and the transform gate. Default `true`
- `init_kernel`: initializer for the input to hidden weights
- `bias`: include a bias or not. Default is `true`

# Equations
```math
\begin{aligned}
s_{\ell}^{[t]} &= h_{\ell}^{[t]} \odot t_{\ell}^{[t]} + s_{\ell-1}^{[t]} \odot c_{\ell}^{[t]}, \\
\text{where} \\
h_{\ell}^{[t]} &= \tanh(W_h x^{[t]}\mathbb{I}_{\ell = 1} + U_{h_{\ell}} s_{\ell-1}^{[t]} + b_{h_{\ell}}), \\
t_{\ell}^{[t]} &= \sigma(W_t x^{[t]}\mathbb{I}_{\ell = 1} + U_{t_{\ell}} s_{\ell-1}^{[t]} + b_{t_{\ell}}), \\
c_{\ell}^{[t]} &= \sigma(W_c x^{[t]}\mathbb{I}_{\ell = 1} + U_{c_{\ell}} s_{\ell-1}^{[t]} + b_{c_{\ell}})
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
    layers = []
    for layer in 1:depth
        if layer == 1
            real_in = input_size + hidden_size
        else
            real_in = hidden_size
        end
        rhn = RHNCellUnit(real_in => hidden_size; cell_kwargs...)
        push!(layers, rhn)
    end
    return RHNCell(Chain(layers), couple_carry)
end

function initialstates(rhn::RHNCell)
    return initialstates(first(rhn.layers))
end

function (rhn::RHNCell)(inp::AbstractArray)
    state = initialstates(rhn)
    return rhn(inp, state)
end

function (rhn::RHNCell)(inp::AbstractArray, state::AbstractVecOrMat)
    current_state = colify(state)

    for (i, layer) in enumerate(rhn.layers.layers)
        if i == 1
            inp_combined = vcat(inp, current_state)
        else
            inp_combined = current_state
        end

        pre_h, pre_t, pre_c = layer(inp_combined)

        # Apply nonlinearities
        hidden_gate = tanh_fast.(pre_h)
        transform_gate = sigmoid_fast.(pre_t)
        carry_gate = sigmoid_fast.(pre_c)

        # Highway component
        if rhn.couple_carry
            current_state = (hidden_gate .- current_state) .* transform_gate .+
                            current_state
        else
            current_state = hidden_gate .* transform_gate .+ current_state .* carry_gate
        end
    end

    return current_state, current_state
end

# TODO fix implementation here
@doc raw"""
    RHN(input_size => hidden_size, [depth];
        return_state = false,
        kwargs...)

[Recurrent highway network](https://arxiv.org/pdf/1607.03474).
See [`RHNCellUnit`](@ref) for a the unit component of this layer.
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
s_{\ell}^{[t]} &= h_{\ell}^{[t]} \odot t_{\ell}^{[t]} + s_{\ell-1}^{[t]} \odot c_{\ell}^{[t]}, \\
\text{where} \\
h_{\ell}^{[t]} &= \tanh(W_h x^{[t]}\mathbb{I}_{\ell = 1} + U_{h_{\ell}} s_{\ell-1}^{[t]} + b_{h_{\ell}}), \\
t_{\ell}^{[t]} &= \sigma(W_t x^{[t]}\mathbb{I}_{\ell = 1} + U_{t_{\ell}} s_{\ell-1}^{[t]} + b_{t_{\ell}}), \\
c_{\ell}^{[t]} &= \sigma(W_c x^{[t]}\mathbb{I}_{\ell = 1} + U_{c_{\ell}} s_{\ell-1}^{[t]} + b_{c_{\ell}})
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

function functor(rhn::RHN{S}) where {S}
    params = (cell=rhn.cell,)
    reconstruct = p -> RHN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function colify(x::AbstractArray)
    # If x is already 2D (e.g. (N,1)), leave it.
    # If x is 1D (N,), reshape to (N, 1).
    ndims(x) == 1 && return reshape(x, (length(x), 1))
    return x
end
