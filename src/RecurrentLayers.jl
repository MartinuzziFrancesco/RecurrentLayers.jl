module RecurrentLayers

using Flux
using Compat: @compat #for @compat public
import Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like
import Flux: glorot_uniform
import Flux: initialstates, scan

abstract type AbstractRecurrentCell end
abstract type AbstractDoubleRecurrentCell <: AbstractRecurrentCell end

function initialstates(rcell::AbstractRecurrentCell)
    return zeros_like(rcell.Wh, size(rcell.Wh, 2))
end

function initialstates(rcell::AbstractDoubleRecurrentCell)
    state = zeros_like(rcell.Wh, size(rcell.Wh, 2))
    second_state = zeros_like(rcell.Wh, size(rcell.Wh, 2))
    return state, second_state
end

function (rcell::AbstractRecurrentCell)(inp::AbstractVecOrMat)
    state = initialstates(rcell)
    return rcell(inp, state)
end

abstract type AbstractRecurrentLayer end

function initialstates(rlayer::AbstractRecurrentLayer)
    return initialstates(rlayer.cell)
end

function (rlayer::AbstractRecurrentLayer)(inp::AbstractVecOrMat)
    state = initialstates(rlayer)
    return rlayer(inp, state)
end

function (rlayer::AbstractRecurrentLayer)(
    inp::AbstractArray,
    state::Union{AbstractVecOrMat, Tuple{AbstractVecOrMat, AbstractVecOrMat}})
    @assert ndims(inp) == 2 || ndims(inp) == 3
    return scan(rlayer.cell, inp, state)
end

export MGUCell, LiGRUCell, IndRNNCell, RANCell, LightRUCell, RHNCell,
RHNCellUnit, NASCell, MUT1Cell, MUT2Cell, MUT3Cell, SCRNCell, PeepholeLSTMCell,
FastRNNCell, FastGRNNCell

export MGU, LiGRU, IndRNN, RAN, LightRU, NAS, RHN, MUT1, MUT2, MUT3,
SCRN, PeepholeLSTM, FastRNN, FastGRNN

export StackedRNN

@compat(public, (initialstates))

include("cells/mgu_cell.jl")
include("cells/ligru_cell.jl")
include("cells/indrnn_cell.jl")
include("cells/ran_cell.jl")
include("cells/lightru_cell.jl")
include("cells/rhn_cell.jl")
include("cells/nas_cell.jl")
include("cells/mut_cell.jl")
include("cells/scrn_cell.jl")
include("cells/peepholelstm_cell.jl")
include("cells/fastrnn_cell.jl")

include("wrappers/stackedrnn.jl")

end #module