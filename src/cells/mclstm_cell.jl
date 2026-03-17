"""

"""
struct MCLSTMCell{I, H, V, W, A} <: AbstractDoubleRecurrentCell
    weight_ih::I
    weight_hh::H
    bias_ih::V
    bias_hh::W
    integration_fn::A
end

@layer MCLSTMCell

function MCLSTMCell((input_size, hidden_size)::Pair{<:Int, <:Int};
        init_kernel=glorot_uniform, init_recurrent_kernel=glorot_uniform,
        bias::Bool=true, recurrent_bias::Bool=true,
        integration_mode::Symbol=:addition,
        independent_recurrence::Bool=false)
    weight_ih = init_kernel(5 * hidden_size, input_size)
    weight_hh = _indrec_matrix(independent_recurrence, init_recurrent_kernel, hidden_size, 5)
    bias_ih = create_bias(weight_ih, bias, size(weight_ih, 1))
    bias_hh = create_bias(weight_hh, recurrent_bias, size(weight_hh, 1))
    integration_fn = _integration_fn(integration_mode)
    return MCLSTMCell(weight_ih, weight_hh, bias_ih, bias_hh, integration_fn)
end

function (lstm::MCLSTMCell)(inp::AbstractVecOrMat, (v_state, c_state))
    _size_check(lstm, inp, 1 => size(lstm.weight_ih, 2))
    proj_ih = dense_proj(lstm.weight_ih, inp, lstm.bias_ih)
    proj_hh = dense_proj(lstm.weight_hh, v_state, lstm.bias_hh)
    gates = lstm.integration_fn(proj_ih, proj_hh)
    fg, ig, og, mg, cell = chunk(gates, 5; dims=1)
    new_cstate = @. fg * c_state + ig * cell
    new_state = og .* tanh_fast.(new_cstate)
    new_vstate = mg .* tanh_fast.(new_cstate)
    return new_state, (new_vstate, new_cstate)
end

function initialstates(lstm::MCLSTMCell)
    state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 5)
    second_state = zeros_like(lstm.weight_hh, size(lstm.weight_hh, 1) ÷ 5)
    return state, second_state
end

function Base.show(io::IO, lstm::MCLSTMCell)
    print(io, "MCLSTMCell(", size(lstm.weight_ih, 2),
        " => ", size(lstm.weight_ih, 1) ÷ 5, ")")
end


"""
"""
struct MCLSTM{S, M} <: AbstractRecurrentLayer{S}
    cell::M
end

@layer :noexpand MCLSTM

function MCLSTM((input_size, hidden_size)::Pair{<:Int, <:Int};
        return_state::Bool=false, kwargs...)
    cell = MCLSTMCell(input_size => hidden_size; kwargs...)
    return MCLSTM{return_state, typeof(cell)}(cell)
end

function functor(rnn::MCLSTM{S}) where {S}
    params = (cell=rnn.cell,)
    reconstruct = p -> MCLSTM{S, typeof(p.cell)}(p.cell)
    return params, reconstruct
end

function Base.show(io::IO, lstm::MCLSTM)
    print(io, "MCLSTM(", size(lstm.cell.weight_ih, 2),
        " => ", size(lstm.cell.weight_ih, 1) ÷ 5)
    print(io, ")")
end