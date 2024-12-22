#https://arxiv.org/pdf/1412.7753
struct SCRNCell{I,H,C,V,A} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wc::C
    bias::V
    alpha::A
end

Flux.@layer SCRNCell


@doc raw"""
    SCRNCell((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true,
        alpha = 0.0)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).
See [`SCRN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `alpha`: structural contraint. Default is 0.0

# Equations
```math
\begin{aligned}
s_t &= (1 - \alpha) W_s x_t + \alpha s_{t-1}, \\
h_t &= \sigma(W_h s_t + U_h h_{t-1} + b_h), \\
y_t &= f(U_y h_t + W_y s_t)
\end{aligned}
```

# Forward

    scrncell(inp, (state, cstate))
    scrncell(inp)

## Arguments

- `inp`: The input to the scrncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the SCRNCell.
  They should be vectors of size `hidden_size` or matrices of size `hidden_size x batch_size`.
  If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (new_state, new_cstate)` is the new hidden and cell state. 
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
function SCRNCell((input_size, hidden_size)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias::Bool = true,
    alpha = 0.f0)

    Wi = init_kernel(2 * hidden_size, input_size)
    Wh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    Wc = init_recurrent_kernel(2 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return SCRNCell(Wi, Wh, Wc, b, alpha)
end

function (scrn::SCRNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(scrn, inp, 1 => size(scrn.Wi,2))
    Wi, Wh, Wc, b = scrn.Wi, scrn.Wh, scrn.Wc, scrn.bias

    #split
    gxs = chunk(Wi * inp, 2; dims=1)
    ghs = chunk(Wh, 2; dims=1)
    gcs = chunk(Wc * c_state .+ b, 2; dims=1)

    #compute
    context_layer = (1.f0 .- scrn.alpha) .* gxs[1] .+ scrn.alpha .* c_state
    hidden_layer = sigmoid_fast(gxs[2] .+ ghs[1] * state .+ gcs[1])
    new_state = tanh_fast(ghs[2] * hidden_layer .+ gcs[2])
    return new_state, (new_state, context_layer)
end

Base.show(io::IO, scrn::SCRNCell) =
    print(io, "SCRNCell(", size(scrn.Wi, 2), " => ", size(scrn.Wi, 1)÷3, ")")


struct SCRN{M} <: AbstractRecurrentLayer
    cell::M
end
  
Flux.@layer :noexpand SCRN

@doc raw"""
    SCRN((input_size => hidden_size)::Pair;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true,
        alpha = 0.0)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).
See [`SCRNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `alpha`: structural contraint. Default is 0.0

# Equations
```math
\begin{aligned}
s_t &= (1 - \alpha) W_s x_t + \alpha s_{t-1}, \\
h_t &= \sigma(W_h s_t + U_h h_{t-1} + b_h), \\
y_t &= f(U_y h_t + W_y s_t)
\end{aligned}
```

# Forward

    scrn(inp, (state, cstate))
    scrn(inp)

## Arguments
- `inp`: The input to the scrn. It should be a vector of size `input_size x len`
  or a matrix of size `input_size x len x batch_size`.
- `(state, cstate)`: A tuple containing the hidden and cell states of the SCRN. 
  They should be vectors of size `hidden_size` or matrices of size `hidden_size x batch_size`.
  If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
"""
function SCRN((input_size, hidden_size)::Pair; kwargs...)
    cell = SCRNCell(input_size => hidden_size; kwargs...)
    return SCRN(cell)
end