module RecurrentLayers

using Flux
import Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like
import Flux: glorot_uniform

export MGUCell, LiGRUCell, IndRNNCell, RANCell, LightRUCell, RHNCell,
RHNCellUnit, NASCell, MUT1Cell, MUT2Cell, MUT3Cell, SCRNCell
export MGU, LiGRU, IndRNN, RAN, LightRU, NAS, RHN, MUT1, MUT2, MUT3,
SCRN


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

end #module