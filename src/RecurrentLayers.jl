module RecurrentLayers

using Compat: @compat
using Flux: _size_check, _match_eltype, chunk, create_bias,
    zeros_like, glorot_uniform, scan, @layer,
    default_rng, Chain, Dropout
import Flux: initialstates
import Functors: functor
using NNlib: fast_act, sigmoid_fast, tanh_fast, relu

export MGUCell, LiGRUCell, IndRNNCell, RANCell, LightRUCell, RHNCell,
    RHNCellUnit, NASCell, MUT1Cell, MUT2Cell, MUT3Cell, SCRNCell, PeepholeLSTMCell,
    FastRNNCell, FastGRNNCell
export MGU, LiGRU, IndRNN, RAN, LightRU, NAS, RHN, MUT1, MUT2, MUT3,
    SCRN, PeepholeLSTM, FastRNN, FastGRNN
export StackedRNN

@compat(public, (initialstates))

include("generics.jl")

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


### fallbacks for functors ###
rlayers = (:FastRNN, :FastGRNN, :IndRNN, :LightRU, :LiGRU, :MGU, :MUT1,
    :MUT2, :MUT3, :NAS, :PeepholeLSTM, :RAN, :SCRN)

rcells = (:FastRNNCell, :FastGRNNCell, :IndRNNCell, :LightRUCell, :LiGRUCell,
    :MGUCell, :MUT1Cell, :MUT2Cell, :MUT3Cell, :NASCell, :PeepholeLSTMCell,
    :RANCell, :SCRNCell)

for (rlayer,rcell) in zip(rlayers, rcells)
    @eval begin
        function ($rlayer)(rc::$rcell; return_state::Bool = false)
            return $rlayer{return_state, typeof(rc)}(rc)
        end

        # why wont' this work?
        #function functor(rl::$rlayer{S}) where {S}
        #    params = (cell = rl.cell)
        #    reconstruct = p -> $rlayer{S, typeof(p.cell)}(p.cell)
        #    return params, reconstruct
        #end
    end
end

end #module