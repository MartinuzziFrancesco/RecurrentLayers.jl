#https://icml.cc/2011/papers/524_icmlpaper.pdf
@doc raw"""
    Multiplicative(cell, inp, state)

[Multiplicative RNN](https://icml.cc/2011/papers/524_icmlpaper.pdf). Wraps
a given `cell`, and performs the following forward pass

```math
\begin{aligned}
    \mathbf{m}_t   &= (\mathbf{W}_{mx} \mathbf{x}_t) \circ (\mathbf{W}_{mh} \mathbf{h}_{t-1}), \\
    \mathbf{h}_{t} &= \text{cell}(\mathbf{x}_t, \mathbf{m}_t).
\end{aligned}
```

## Arguments

## Keyword arguments
"""
struct Multiplicative{C, M, H}
    cell::C
    Wm::M
    Wh::H
end

@layer Multiplicative

function initialstates(mrnn::Multiplicative)
    return initialstates(mrnn.cell)
end

function Multiplicative(rcell, (input_size, hidden_size)::Pair{<:Int, <:Int}, args...;
        init_multiplicative_kernel = glorot_uniform,
        init_multiplicativerecurrent_kernel = glorot_uniform, kwargs...)
    cell = rcell(input_size => hidden_size, args...; kwargs...)
    Wm = init_multiplicative_kernel(hidden_size, input_size)
    Wh = init_multiplicativerecurrent_kernel(hidden_size, hidden_size)
    return Multiplicative(cell, Wm, Wh)
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat, state::AbstractVecOrMat)
    m_state = (mrnn.Wm * inp) .* (mrnn.Wh * state)
    new_state = mrnn.cell(inp, m_state)
    return new_state, new_state
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat, (state, c_state))
    m_state = (mrnn.WM * inp) .* (mrnn.Wh * state)
    new_state = mrnn.cell(inp, (m_state, c_state))
    return new_state, (new_state, c_state)
end

function (mrnn::Multiplicative)(inp::AbstractVecOrMat)
    state = initialstates(mrnn)
    return mrnn(inp, state)
end