# based on https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/
struct StackedRNN{L,D,S}
    layers::L
    droput::D
    states::S
end

Flux.@layer StackedRNN trainable=(layers)

"""
    StackedRNN(rlayer, (input_size, hidden_size), args...;
        num_layers = 1, kwargs...)

Constructs a stack of recurrent layers given the recurrent layer type.

Arguments:
  - `rlayer`: Any recurrent layer such as [MGU](@ref), [RHN](@ref), etc... or
    [Flux.RNN](@extref), [Flux.LSTM](@extref), etc... Additionally anything wrapped in
    [Flux.recurrence](@extref) can be used as `rlayer`.
  - `input_size`: Defines the input dimension for the first layer.
  - `hidden_size`: defines the dimension of the hidden layer.
  - `num_layers`: The number of layers to stack. Default is 1.
  - `args...`: Additional positional arguments passed to the recurrent layer.
  - `kwargs...`: Additional keyword arguments passed to the recurrent layers.

Returns:
  A `StackedRNN` instance containing the specified number of RNN layers and their initial states.
"""
function StackedRNN(rlayer, (input_size, hidden_size)::Pair, args...;
    num_layers::Int = 1,
    dropout::Number = 0.0,
    kwargs...)
    layers = []
    for (idx,layer) in enumerate(num_layers)
        in_size = idx == 1 ? input_size : hidden_size
        push!(layers, rlayer(in_size => hidden_size, args...; kwargs...))
    end
    states = [initialstates(layer) for layer in layers]

    return StackedRNN(layers, Dropout(dropout), states)
end

function (stackedrnn::StackedRNN)(inp::AbstractArray)
    for (idx,(layer, state)) in enumerate(zip(stackedrnn.layers, stackedrnn.states))
        inp = layer(inp, state0)
        if !(idx == length(stackedrnn.layers))
            inp = stackedrnn.dropout(inp)
        end
    end
    return inp
end
