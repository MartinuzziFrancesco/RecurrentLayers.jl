#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60

struct RHNCellUnit{I,V}
    weight::I
    bias::V
end

Flux.@layer RHNCellUnit

"""
    RHNCellUnit((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        bias = true)
"""
function RHNCellUnit((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    bias = true)
    weight = init_kernel(3 * hidden_size, input_size)
    b = create_bias(weight, bias, size(weight, 1))
    return RHNCellUnit(weight, b)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat)
    state = zeros_like(inp, size(rhn.weight, 2))
    return rhn(inp, state)
end

function (rhn::RHNCellUnit)(inp::AbstractVecOrMat, state)
    _size_check(rhn, inp, 1 => size(rhn.weight, 2))
    weight, bias = rhn.weight, rhn.bias

    #compute
    pre_nonlin = weight * inp .+ bias

    #split
    pre_h, pre_t, pre_c = chunk(pre_nonlin, 3, dims = 1)
    return pre_h, pre_t, pre_c
end

Base.show(io::IO, rhn::RHNCellUnit) =
    print(io, "RHNCellUnit(", size(rhn.weight, 2), " => ", size(rhn.weight, 1)÷3, ")")

struct RHNCell{C}
    layers::C
    couple_carry::Bool
end

Flux.@layer RHNCell

@doc raw"""
    RHNCell((input_size => hidden_size), depth=3;
        couple_carry::Bool = true,
        cell_kwargs...)

[Recurrent highway network](https://arxiv.org/pdf/1607.03474).
See [`RHNCellUnit`](@ref) for a the unit component of this layer.
See [`RHN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `depth`: depth of the recurrence. Default is 3
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
function RHNCell((input_size, hidden_size), depth::Int = 3;
    couple_carry::Bool = true, #sec 5, setup
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


function (rhn::RHNCell)(inp, state=nothing)

    #not ideal
    if state == nothing
        state = zeros_like(inp, size(rhn.layers.layers[2].weight, 2))
    end

    current_state = state

    for (i, layer) in enumerate(rhn.layers.layers)
        if i == 1
            inp_combined = vcat(inp, current_state)
        else
            inp_combined = current_state
        end

        pre_h, pre_t, pre_c = layer(inp_combined)

        # Apply nonlinearities
        hidden_gate = tanh.(pre_h)
        transform_gate = sigmoid.(pre_t)
        carry_gate = sigmoid.(pre_c)

        # Highway component
        if rhn.couple_carry
            current_state = (hidden_gate .- current_state) .* transform_gate .+ current_state
        else
            current_state = hidden_gate .* transform_gate .+ current_state .* carry_gate
        end
    end

    return current_state, current_state
end

# TODO fix implementation here
struct RHN{M}
    cell::M
end
  
Flux.@layer :expand RHN

@doc raw"""
    RHN((input_size => hidden_size)::Pair depth=3; kwargs...)

[Recurrent highway network](https://arxiv.org/pdf/1607.03474).
See [`RHNCellUnit`](@ref) for a the unit component of this layer.
See [`RHNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `depth`: depth of the recurrence. Default is 3
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
"""
function RHN((input_size, hidden_size)::Pair, depth=3; kwargs...)
    cell = RHNCell(input_size => hidden_size, depth; kwargs...)
    return RHN(cell)
end
  
function (rhn::RHN)(inp)
    state = zeros_like(inp, size(rhn.cell.layers[2].weights, 2))
    return rhn(inp, state)
end
  
function (rhn::RHN)(inp, state)
    @assert ndims(inp) == 2 || ndims(inp) == 3
    new_state = []
    for inp_t in eachslice(inp, dims=2)
        state = rhn.cell(inp_t, state)
        new_state = vcat(new_state, [state])
    end
    return stack(new_state, dims=2)
end