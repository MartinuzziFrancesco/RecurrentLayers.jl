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

abstract type AbstractRecurrentLayer{S} end

function initialstates(rlayer::AbstractRecurrentLayer)
    return initialstates(rlayer.cell)
end

function (rlayer::AbstractRecurrentLayer)(inp::AbstractVecOrMat)
    state = initialstates(rlayer)
    return rlayer(inp, state)
end

function (rlayer::AbstractRecurrentLayer{false})(inp::AbstractArray,
        state::Union{AbstractVecOrMat, Tuple{AbstractVecOrMat, AbstractVecOrMat}})
    @assert ndims(inp) == 2 || ndims(inp) == 3
    @assert typeof(state)==typeof(initialstates(rlayer)) """\n
       The layer $rlayer is calling states not supported by its
       forward method. Check if this is a single or double return
       recurrent layer, and adjust your inputs accordingly.
    """
    return first(scan(rlayer.cell, inp, state))
end

function (rlayer::AbstractRecurrentLayer{true})(inp::AbstractArray,
        state::Union{AbstractVecOrMat, Tuple{AbstractVecOrMat, AbstractVecOrMat}})
    @assert ndims(inp) == 2 || ndims(inp) == 3
    @assert typeof(state)==typeof(initialstates(rlayer)) """\n
       The layer $rlayer is calling states not supported by its
       forward method. Check if this is a single or double return
       recurrent layer, and adjust your inputs accordingly.
    """
    return scan(rlayer.cell, inp, state)
end
