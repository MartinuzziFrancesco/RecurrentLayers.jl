module RecurrentLayers

using Flux
import Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like
import Flux: glorot_uniform

export MGUCell, LiGRUCell, IndRNNCell, RANCell, LightRUCell, RHNCell, RHNCellUnit
export MGU, LiGRU, IndRNN, RAN, LightRU

include("mgu_cell.jl")
include("ligru_cell.jl")
include("indrnn_cell.jl")
include("ran_cell.jl")
include("lru_cell.jl")
include("rhn_cell.jl")

end #module
