#https://arxiv.org/abs/1705.08639
@doc raw"""
    FSRNNCell(input_size => hidden_size,
        fast_cells, slow_cell)

Fast slow recurrent neural network cell [^Mujika2017].
See [`FSRNN`](@ref) for a layer that processes entire sequences.

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

[^Mujika2017]: Mujika, A. et al.  
    _Fast-Slow Recurrent Neural Networks._  
    NeurIPS 2017.
"""
struct FSRNNCell{F, S} <: AbstractRecurrentCell
    fast_cells::F
    slow_cell::S
end

@layer FSRNNCell

function FSRNNCell((input_size, hidden_size)::Pair{<:Int, <:Int},
        fast_cells, slow_cell)
    @assert length(fast_cells) > 1
    f_cells = []
    for (cell_idx, fast_cell) in enumerate(fast_cells)
        in_size = cell_idx == 1 ? input_size : hidden_size
        push!(f_cells, fast_cell(in_size => hidden_size))
    end
    s_cell = slow_cell(hidden_size => hidden_size)
    return FSRNNCell(f_cells, s_cell)
end

function initialstates(fsrnn::FSRNNCell)
    fast_state = initialstates(first(fsrnn.fast_cells))
    slow_state = initialstates(fsrnn.slow_cell)
    return fast_state, slow_state
end

function (fsrnn::FSRNNCell)(inp::AbstractVecOrMat, (fast_state, slow_state))
    for (cell_idx, fast_cell) in enumerate(fsrnn.fast_cells)
        inp, fast_state = fast_cell(inp, fast_state)
        if cell_idx == 1
            inp, slow_state = fsrnn.slow_cell(inp, slow_state)
        end
    end
    return inp, (fast_state, slow_state)
end

function Base.show(io::IO, fsrnn::FSRNNCell)
    print(io, "FSRNNCell(", size(first(fsrnn.fast_cells).Wi, 2), " => ",
        size(first(fsrnn.fast_cells).Wi, 1) ÷ 4, ")")
end

@doc raw"""
    FSRNN(input_size => hidden_size,
        fast_cells, slow_cell;
        return_state=false)

Fast slow recurrent neural network [^Mujika2017].
See [`FSRNNCell`](@ref) for a layer that processes a single sequence.

# Arguments
- `input_size => hidden_size`: input and inner dimension of the layer.
- `fast_cells`: a vector of the fast cells. Must be minimum of length 2.
- `slow_cell`: the chosen slow cell.
- `return_state`: option to return the last state. Default is `false`.

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

    fsrnn(inp, (fast_state, slow_state))
    fsrnn(inp)

## Arguments

- `inp`: The input to the fsrnn. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `(fast_state, slow_state)`: A tuple containing the hidden and cell states of the FSRNN.
  They should be vectors of size `hidden_size` or matrices of size
  `hidden_size x batch_size`. If not provided, they are assumed to be vectors of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- New hidden states `new_states` as an array of size `hidden_size x len x batch_size`.
  When `return_state = true` it returns a tuple of the hidden stats `new_states` and
  the last state of the iteration.

[^Mujika2017]: Mujika, A. et al.  
    _Fast-Slow Recurrent Neural Networks._  
    NeurIPS 2017.
"""
struct FSRNN{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand FSRNN

function FSRNN((input_size, hidden_size)::Pair{<:Int, <:Int},
        fast_cells, slow_cell; return_state::Bool=false)
    cell = FSRNNCell(input_size => hidden_size, fast_cells, slow_cell)
    return FSRNN{return_state, typeof(cell)}(cell)
end

function functor(fsrnn::FSRNN{S}) where {S}
    params = (cell=fsrnn.cell,)
    reconstruct = p -> FSRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, fsrnn::FSRNN)
    print(io, "FSRNN(", size(first(fsrnn.cell.fast_cells).Wi, 2),
        " => ", size(first(fsrnn.cell.fast_cells).Wi, 1))
    print(io, ")")
end
