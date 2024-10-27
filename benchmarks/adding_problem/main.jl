using Flux
using RecurrentLayers
using MLUtils
using StatsBase

function generate_adding_data(
    sequence_length::Int,
    n_samples::Int,
    return_dataloader::Bool=true,
    batchsize::Int=64,
    shuffle::Bool=true
)
    random_sequence = rand(Float32, n_samples, sequence_length, 1)
    mask_sequence = zeros(Float32, n_samples, sequence_length, 1)
    targets = zeros(n_samples, 1)

    for i in 1:n_samples
        idxs = sample(1:sequence_length, 2; replace=false)
        mask_sequence[i, idxs, 1] = 1
        targets[i] = sum(random_sequence[i, idx, 1])
    end

    inputs = cat(random_sequence, mask_sequence, dims=3)

    if return_dataloader
        dataloader = DataLoader(
            (data=inputs, label=targets),
            batchsize = batchsize,
            shuffle=shuffle
        )
        return dataloader
    else
        return inputs, targets
    end
end