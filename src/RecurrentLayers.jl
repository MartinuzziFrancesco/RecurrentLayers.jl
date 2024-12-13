module RecurrentLayers

using Flux
import Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like
import Flux: glorot_uniform
import Flux: initialstates

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
    return rcell(inp, state)
end

export MGUCell, LiGRUCell, IndRNNCell, RANCell, LightRUCell, RHNCell,
RHNCellUnit, NASCell, MUT1Cell, MUT2Cell, MUT3Cell, SCRNCell, PeepholeLSTMCell,
FastRNNCell, FastGRNNCell
export MGU, LiGRU, IndRNN, RAN, LightRU, NAS, RHN, MUT1, MUT2, MUT3,
SCRN, PeepholeLSTM, FastRNN, FastGRNN


#TODO add double bias
include("mgu_cell.jl")
include("ligru_cell.jl")
include("indrnn_cell.jl")
include("ran_cell.jl")
include("lightru_cell.jl")
include("rhn_cell.jl")
include("nas_cell.jl")
include("mut_cell.jl")
include("scrn_cell.jl")
include("peepholelstm_cell.jl")
include("fastrnn_cell.jl")

end #module