#https://arxiv.org/abs/1705.08639
struct FSRNNCell{F,S} <: AbstractRecurrentCell
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

function (fsrnn::FSRNNCell)(inp::AbstractArray, (fast_state, slow_state))
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

struct FSRNN{S,M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand FSRNN

function FSRNN((input_size, hidden_size)::Pair{<:Int, <:Int},
        fast_cells, slow_cell; return_state::Bool=false)
    cell = FSRNNCell(input_size => hidden_size, fast_cells, slow_cell)
    return FSRNN{return_state, typeof(cell)}(cell)
end

function functor(rnn::FSRNN{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> FSRNN{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, fsrnn::FSRNN)
    print(io, "FSRNN(", size(first(fsrnn.cell.fast_cells).Wi, 2),
        " => ", size(first(fsrnn.cell.fast_cells).Wi, 1))
    print(io, ")")
end