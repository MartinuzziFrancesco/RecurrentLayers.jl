# based on https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/
struct StackedRNN{L,D,S}
    layers::L
    dropout::D
    states::S
end

@layer StackedRNN trainable=(layers)

@doc raw"""
    StackedRNN(rlayer, (input_size, hidden_size), args...;
        num_layers = 1, kwargs...)

Constructs a stack of recurrent layers given the recurrent layer type.

Arguments:
  - `rlayer`: Any recurrent layer such as [MGU](@ref), [RHN](@ref), etc... or
    [`Flux.RNN`](@extref), [`Flux.LSTM`](@extref), etc.
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
    dims = :,
    active::Union{Bool,Nothing} = nothing,
    rng = default_rng(),
    kwargs...)
    #build container
    layers = []
    #warn for dropout and num_layers
    if num_layers ==1 && dropout != 0.0
        @warn("Dropout is not applied when num_layers is 1.")
    end

    for idx in 1:num_layers
        in_size = idx == 1 ? input_size : hidden_size
        push!(layers, rlayer(in_size => hidden_size, args...; kwargs...))
    end
    states = [initialstates(layer) for layer in layers]

    return StackedRNN(layers,
        Dropout(dropout; dims = dims, active = active, rng = rng),
        states)
end

function (stackedrnn::StackedRNN)(inp::AbstractArray)
    @assert length(stackedrnn.layers) == length(stackedrnn.states) "Mismatch in layers vs. states length!"
    @assert !isempty(stackedrnn.layers) "StackedRNN has no layers!"
    for idx in eachindex(stackedrnn.layers)
        inp = stackedrnn.layers[idx](inp, stackedrnn.states[idx])
        if idx != length(stackedrnn.layers)
            inp = stackedrnn.dropout(inp)
        end
    end
    return inp
end
