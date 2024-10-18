module RecurrentLayers

using Flux
import Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like
import Flux: glorot_uniform

export MGUCell, LiGRUCell, IndRNNCell
export MGU, LiGRU, IndRNN

include("mgu_cell.jl")
include("ligru_cell.jl")
include("indrnn_cell.jl")

end #module
