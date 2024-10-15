module RecurrentLayers

using Flux
import Flux: _size_check, _match_eltype, multigate, reshape_cell_output
import Flux: glorot_uniform

export MGUCell

include("mgu_cell.jl")

end #module
