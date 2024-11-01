function _process_layer(layer_input, state, cell)
    new_states = map(eachslice(layer_input, dims=2)) do inp_t
        state = cell(inp_t, state)
        state
    end

    layer_output = stack(new_states, dims=2)
    return layer_output, state
end