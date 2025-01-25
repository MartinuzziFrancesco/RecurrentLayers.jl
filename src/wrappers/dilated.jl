using Flux

"""
    Dilated(cell, inputs, rate)

Dilated RNN layer with a single recurrent cell. 

* `cell` is any Flux recurrent cell (e.g. `MGUCell`, `LSTMCell`, etc.).
* `inputs` is a Vector of length `n_steps`, where each element is the data at one time step.
  For instance, `inputs[t]` might be a `(input_dim, batch_size)` matrix.
* `rate` is the dilation rate.

"""
using Flux

struct Dilated{L,D,S,R}
    layers::L
    dropout::D
    states::S
    rates::R
end

@layer Dilated trainable=(layers)

function Dilated(rlayer::Function, (input_size, hidden_size)::Pair{<:Int,<:Int};
        num_layers::Int=1, rates::Union{Int,AbstractVector{<:Int}}=1, dropout::Real=0.0,
        dims=:, active::Union{Bool,Nothing}=nothing, rng=Flux.default_rng(), kwargs...)
    if rates isa Int
        rates = fill(rates, num_layers)
    end
    layers = Vector{Any}(undef, num_layers)
    for i in 1:num_layers
        in_size = (i == 1) ? input_size : hidden_size
        layers[i] = rlayer(in_size => hidden_size; kwargs...)
    end
    drop = Dropout(dropout; dims=dims, active=active, rng=rng)
    states = [initialstates(layers[i]) for i in 1:num_layers]
    Dilated(layers, drop, states, rates)
end

function _dilated_forward(cell::AbstractRecurrentCell, state, inputs::Vector{<:AbstractMatrix}, rate::Int)
    n_steps = length(inputs)
    even = (n_steps % rate) == 0
    original_n_steps = n_steps
    if !even
        dilated_n_steps = div(n_steps, rate) + 1
        pad_len = dilated_n_steps * rate - n_steps
        zero_tensor = fill(0.0, size(inputs[1]))
        for _ in 1:pad_len
            push!(inputs, zero_tensor)
        end
    else
        dilated_n_steps = div(n_steps, rate)
    end
    new_inputs = Vector{AbstractMatrix}(undef, dilated_n_steps)
    idx = 1
    for i in 1:dilated_n_steps
        chunk = inputs[idx : idx + rate - 1]
        idx += rate
        new_inputs[i] = vcat(chunk...)
    end
    d_outputs = Vector{AbstractMatrix}(undef, dilated_n_steps)
    hidden_state = state
    for i in 1:dilated_n_steps
        inp_t = new_inputs[i]'
        out_t, hidden_state = cell(inp_t, hidden_state)
        d_outputs[i] = out_t'
    end
    splitted = map(o -> [
        o[(j-1)*size(inputs[1],1)+1 : j*size(inputs[1],1), :]
        for j in 1:rate
    ], d_outputs)
    splitted_flat = reduce(vcat, splitted)
    final_outputs = splitted_flat[1:original_n_steps]
    return final_outputs, hidden_state
end

function (drnn::DilatedRNN)(inp::Vector{<:AbstractMatrix})
    x = inp
    for i in 1:length(drnn.layers)
        cell   = drnn.layers[i]
        state  = drnn.states[i]
        rate   = drnn.rates[i]
        out, new_state = _dilated_forward(cell, state, x, rate)
        drnn.states[i] = new_state
        if i < length(drnn.layers)
            out = drnn.dropout(out)
        end
        x = out
    end
    x
end
