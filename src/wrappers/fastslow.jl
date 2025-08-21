#https://arxiv.org/abs/1705.08639
@doc raw"""
    FastSlow(input_size => hidden_size,
        fast_cells, slow_cell)

Fast slow recurrent neural network cell [Mujika2017](@cite).

# Arguments
- `input_size => hidden_size`: input and inner dimension of the layer
- `fast_cells`: a vector of the fast cells. Must be minimum of length 2.
- `slow_cell`: the chosen slow cell.

# Equations

```math
\begin{aligned}
    \mathbf{h}^{F_1}(t) &= f^{F_1}\left( \mathbf{h}^{F_k}(t-1), \mathbf{x}(t)
        \right), \\
    \mathbf{h}^{S}(t) &= f^{S}\left( \mathbf{h}^{S}(t-1), \mathbf{h}^{F_1}(t)
        \right), \\
    \mathbf{h}^{F_2}(t) &= f^{F_2}\left( \mathbf{h}^{F_1}(t), \mathbf{h}^{S}(t)
        \right), \\
    \mathbf{h}^{F_i}(t) &= f^{F_i}\left( \mathbf{h}^{F_{i-1}}(t) \right) \quad
        \text{for } 3 \leq i \leq k
\end{aligned}
```

# Forward

    fsrnncell(inp, (fast_state, slow_state))
    fsrnncell(inp)

## Arguments

- `inp`: The input to the fsrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(fast_state, slow_state)`: A tuple containing the hidden and cell states of the
  FSRNNCell. They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where `output = new_state` is the new hidden state and
  `state = (fast_state, slow_state)` is the new hidden and cell state.
  They are tensors of size `hidden_size` or `hidden_size x batch_size`.
"""
struct FastSlow{F, S} <: AbstractDoubleRecurrentCell
    fast_cells::F
    slow_cell::S
end

@layer FastSlow

function FastSlow(fast_cells, slow_cell,
        (input_size, hidden_size)::Pair{<:Int, <:Int}, num_fast_cells=2)
    if !(fast_cells isa AbstractVector)
        fast_cells = fill(fast_cells, num_fast_cells)
    end
    f_cells = []
    for (cell_idx, fast_cell) in enumerate(fast_cells)
        in_size = cell_idx == 1 ? input_size : hidden_size
        push!(f_cells, fast_cell(in_size => hidden_size))
    end
    s_cell = slow_cell(hidden_size => hidden_size)
    return FastSlow(f_cells, s_cell)
end

function initialstates(fsrnn::FastSlow)
    fast_state = initialstates(first(fsrnn.fast_cells))
    slow_state = initialstates(fsrnn.slow_cell)
    return fast_state, slow_state
end

function (fsrnn::FastSlow)(inp::AbstractVecOrMat, (fast_state, slow_state))
    re_inp, re_faststate, re_slowstate = inp, fast_state, slow_state
    for (cell_idx, fast_cell) in enumerate(fsrnn.fast_cells)
        re_inp, re_faststate = fast_cell(re_inp, re_faststate)
        if cell_idx == 1
            re_inp, re_slowstate = fsrnn.slow_cell(re_inp, re_slowstate)
        end
    end
    return re_inp, (re_faststate, re_slowstate)
end

function (fsrnn::FastSlow)(inp::AbstractVecOrMat)
    fast_state, slow_state = initialstates(fsrnn)
    return fsrnn(inp, (fast_state, slow_state))
end

function Base.show(io::IO, fsrnn::FastSlow)
    print(io, "FastSlow(", size(first(fsrnn.fast_cells).Wi, 2), " => ",
        size(first(fsrnn.fast_cells).Wi, 1) ÷ 4, ")")
end
