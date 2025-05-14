#https://icml.cc/2011/papers/524_icmlpaper.pdf
@doc raw"""
    Multiplicative(cell, inp, state)

[Multiplicative RNN](https://icml.cc/2011/papers/524_icmlpaper.pdf). Wraps
a given `cell`, and performs the following forward pass.

Currently this wrapper does not support the following cells:

  - `RHNCell`
  - `RHNCellUnit`
  - `FSRNNCell`
  - `TLSTMCell`

```math
\begin{aligned}
    \mathbf{m}_t   &= (\mathbf{W}_{mx} \mathbf{x}_t) \circ (\mathbf{W}_{mh} \mathbf{h}_{t-1}), \\
    \mathbf{h}_{t} &= \text{cell}(\mathbf{x}_t, \mathbf{m}_t).
\end{aligned}
```

## Arguments

  - `rcell`: A recurrent cell constructor such as [MGUCell](@ref), or
   [`Flux.LSTMCell`](@extref) etc.
  - `input_size`: Defines the input dimension for the first layer.
  - `hidden_size`: defines the dimension of the hidden layer.
  - `args...`: positional arguments for the `rcell`.

## Keyword arguments

  - `init_multiplicative_kernel`:Initializer for the multiplicative input kernel.
    Default is glorot_uniform.
  - `init_multiplicativerecurrent_kernel`:Initializer for the multiplicative recurrent
    kernel. Default is glorot_uniform.
  - `kwargs...`: keyword arguments for the `rcell`.

# Forward

    mrnn(inp, state)
    mrnn(inp, (state, c_state))
    mrnn(inp)

## Arguments

  - `inp`: The input to the `rcell`. It should be a vector of size `input_size`
   or a matrix of size `input_size x batch_size`.
  - `state`: The hidden state of the `rcell`, is single return.
    It should be a vector of size `hidden_size` or a matrix of size
    `hidden_size x batch_size`. If not provided, it is assumed to be a
    vector of zeros, initialized by [`Flux.initialstates`](@extref).
  - `(state, cstate)`: A tuple containing the hidden and cell states of the `rcell`.
    if double return. They should be vectors of size `hidden_size` or matrices of size
    `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
    initialized by [`Flux.initialstates`](@extref).

## Returns

Either of 
  - A tuple `(output, state)`, where both elements are given by the updated state
    `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`, if the
    `rcell` is single return (e.g. [`Flux.RNNCell`](@extref)).

  - A tuple `(output, state)`, where `output = new_state` is the new hidden state and
    `state = (new_state, new_cstate)` is the new hidden and cell state. 
    They are tensors of size `hidden_size` or `hidden_size x batch_size`.
    This applies if the `rcell` is double return (e.g. [`Flux.LSTMCell`](@extref)).

# Examples

When used to wrap a cell, `Multiplicative` will behave as the cell wrapped, taking input
data in the same format, and returning states like the `rcell` would.

```jldoctest
julia> using RecurrentLayers

julia> mrnn = Multiplicative(MGUCell, 3 => 5)
Multiplicative(
  5×3 Matrix{Float32},                  # 15 parameters
  5×5 Matrix{Float32},                  # 25 parameters
  MGUCell(3 => 5),                      # 90 parameters
)                   # Total: 5 arrays, 130 parameters, 792 bytes.

```

In order to make `Multiplicative` act on a full sequence it is possible to wrap it
in a [`Flux.Recurrence`](@extref) layer.

```jldoctest
julia> using RecurrentLayers, Flux

julia> wrap = Recurrence(Multiplicative(AntisymmetricRNNCell, 2 => 4))
Recurrence(
  Multiplicative(
    4×2 Matrix{Float32},                # 8 parameters
    4×4 Matrix{Float32},                # 16 parameters
    AntisymmetricRNNCell(2 => 4, tanh),  # 28 parameters
  ),
)                   # Total: 5 arrays, 52 parameters, 488 bytes.

```
"""
struct Multiplicative{M, H, C}
    Wm::M
    Wh::H
    cell::C
end

@layer Multiplicative

function initialstates(mrnn::Multiplicative)
    return initialstates(mrnn.cell)
end

function Multiplicative(rcell, (input_size, hidden_size)::Pair{<:Int, <:Int}, args...;
        init_multiplicative_kernel=glorot_uniform,
        init_multiplicativerecurrent_kernel=glorot_uniform, kwargs...)
    cell = rcell(input_size => hidden_size, args...; kwargs...)
    Wm = init_multiplicative_kernel(hidden_size, input_size)
    Wh = init_multiplicativerecurrent_kernel(hidden_size, hidden_size)
    return Multiplicative(Wm, Wh, cell)
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    m_state = (mrnn.Wm * inp) .* (mrnn.Wh * state)
    return mrnn.cell(inp, m_state)
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat, (state, c_state))
    m_state = (mrnn.Wm * inp) .* (mrnn.Wh * state)
    return mrnn.cell(inp, (m_state, c_state))
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat)
    state = initialstates(mrnn)
    return mrnn(inp, state)
end
