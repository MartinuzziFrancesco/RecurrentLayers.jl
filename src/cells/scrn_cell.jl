#https://arxiv.org/pdf/1412.7753

@doc raw"""
    SCRNCell(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, alpha = 0.0)

Structurally contraint recurrent unit [Mikolov2014](@cite).
See [`SCRN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
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
struct SCRNCell{I, H, C, V, A} <: AbstractDoubleRecurrentCell
    Wi::I
    Wh::H
    Wc::C
    bias::V
    alpha::A
end

@layer SCRNCell

function SCRNCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, alpha=0.0f0)
    Wi = init_kernel(2 * hidden_size, input_size)
    Wh = init_recurrent_kernel(2 * hidden_size, hidden_size)
    Wc = init_recurrent_kernel(2 * hidden_size, hidden_size)
    b = create_bias(Wi, bias, size(Wh, 1))
    return SCRNCell(Wi, Wh, Wc, b, [eltype(Wi)(alpha)])
end

function (scrn::SCRNCell)(inp::AbstractVecOrMat, (state, c_state))
    _size_check(scrn, inp, 1 => size(scrn.Wi, 2))
    Wi, Wh, Wc, b = scrn.Wi, scrn.Wh, scrn.Wc, scrn.bias

    #split
    gxs = chunk(Wi * inp, 2; dims=1)
    ghs = chunk(Wh, 2; dims=1)

    #compute
    t_ones = eltype(Wi)(1.0f0)
    new_cstate = @. (t_ones - scrn.alpha) * gxs[1] + scrn.alpha * c_state
    gcs = chunk(Wc * new_cstate .+ b, 2; dims=1)
    hidden_layer = sigmoid_fast.(gxs[2] .+ ghs[1] * state .+ gcs[1])
    new_state = tanh_fast.(ghs[2] * hidden_layer .+ gcs[2])
    return new_state, (new_state, new_cstate)
end

function Base.show(io::IO, scrn::SCRNCell)
    print(io, "SCRNCell(", size(scrn.Wi, 2), " => ", size(scrn.Wi, 1) ÷ 3, ")")
end

@doc raw"""
    SCRN(input_size => hidden_size;
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, alpha = 0.0,
        return_state = false)

Structurally contraint recurrent unit [Mikolov2014](@cite).
See [`SCRNCell`](@ref) for a layer that processes a single sequence.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.
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
    print(io, "SCRN(", size(scrn.cell.Wi, 2), " => ", size(scrn.cell.Wi, 1))
    print(io, ")")
end
