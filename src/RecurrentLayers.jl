module RecurrentLayers

using Compat: @compat
using Flux: _size_check, _match_eltype, chunk, create_bias, zeros_like, glorot_uniform,
            scan, @layer, default_rng, Chain, Dropout, sigmoid_fast, tanh_fast, relu
import Flux: initialstates
import Functors: functor
using LinearAlgebra: I, transpose
using NNlib: fast_act

export AntisymmetricRNNCell, ATRCell, BRCell, CFNCell, coRNNCell, FastGRNNCell, FastRNNCell,
       FSRNNCell, GatedAntisymmetricRNNCell, IndRNNCell, JANETCell, LEMCell, LiGRUCell,
       LightRUCell, MGUCell, MinimalRNNCell, MUT1Cell, MUT2Cell, MUT3Cell, NASCell, NBRCell,
       PeepholeLSTMCell, RANCell, RHNCell, RHNCellUnit, SCRNCell, SGRNCell, STARCell,
       TGRUCell,
       TLSTMCell, TRNNCell, UnICORNNCell
export AntisymmetricRNN, ATR, BR, CFN, coRNN, FastGRNN, FastRNN, FSRNN,
       GatedAntisymmetricRNN,
       IndRNN, JANET, LEM, LiGRU, LightRU, MGU, MinimalRNN, MUT1, MUT2, MUT3, NAS, NBR,
       PeepholeLSTM, RAN, RHN, SCRN, SGRN, STAR, TGRU, TLSTM, TRNN, UnICORNN
export StackedRNN

@compat(public, (initialstates))

include("generics.jl")

include("cells/antisymmetricrnn_cell.jl")
include("cells/atr_cell.jl")
include("cells/br_cell.jl")
include("cells/cfn_cell.jl")
include("cells/cornn_cell.jl")
include("cells/fastrnn_cell.jl")
include("cells/fsrnn_cell.jl")
include("cells/indrnn_cell.jl")
include("cells/janet_cell.jl")
include("cells/lem_cell.jl")
include("cells/lightru_cell.jl")
include("cells/ligru_cell.jl")
include("cells/mgu_cell.jl")
include("cells/minimalrnn_cell.jl")
include("cells/mut_cell.jl")
include("cells/nas_cell.jl")
include("cells/peepholelstm_cell.jl")
include("cells/ran_cell.jl")
include("cells/rhn_cell.jl")
include("cells/scrn_cell.jl")
include("cells/sgrn_cell.jl")
include("cells/star_cell.jl")
include("cells/trnn_cell.jl")
include("cells/unicornn_cell.jl")

include("wrappers/stackedrnn.jl")

### fallbacks for functors ###
rlayers = (
    :AntisymmetricRNN, :ATR, :BRCell, :CFN, :coRNN, :FastGRNN, :FastRNN, :FSRNN, :IndRNN,
    :JANET, :LEM, :LiGRU, :LightRU, :MGU, :MinimalRNN, :MUT1, :MUT2, :MUT3, :NAS, :NBR,
    :PeepholeLSTM, :RAN, :SCRN, :SGRN, :STAR, :TGRU, :TLSTM, :TRNN, :UnICORNN)

rcells = (
    :AntisymmetricRNNCell, :ATRCell, :BR, :CFNCell, :coRNNCell, :FastGRNNCell, :FastRNNCell,
    :FSRNNCell, :IndRNNCell, :JANETCell, :LEMCell, :LiGRUCell, :LightRUCell,
    :MGUCell, :MinimalRNNCell, :MUT1Cell, :MUT2Cell, :MUT3Cell, :NASCell, :NBRCell,
    :PeepholeLSTMCell, :RANCell, :SCRNCell, :SGRNCell, :STARCell, :TGRUCell, :TLSTMCell,
    :TRNNCell, :UnICORNNCell)

for (rlayer, rcell) in zip(rlayers, rcells)
    @eval begin
        function ($rlayer)(rc::$rcell; return_state::Bool=false)
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
