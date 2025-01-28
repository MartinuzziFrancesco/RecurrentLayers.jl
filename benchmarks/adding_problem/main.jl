using Flux, RecurrentLayers, MLUtils, StatsBase, Comonicon, Printf, CUDA

function generate_adding_data(
        sequence_length::Int,
        n_samples::Int;
        kwargs...
)
    random_sequence = rand(Float32, 1, sequence_length, n_samples)
    mask_sequence = zeros(Float32, 1, sequence_length, n_samples)
    targets = zeros(Float32, n_samples)

    for i in 1:n_samples
        idxs = sample(1:sequence_length, 2; replace=false)
        mask_sequence[1, idxs, i] .= 1
        targets[i] = sum(Float32, random_sequence[1, idxs, i])
    end

    inputs = cat(random_sequence, mask_sequence; dims=1)
    @assert size(inputs, 3) == size(targets, 1)

    dataloader = DataLoader(
        (data=inputs, label=targets);
        kwargs...
    )
    return dataloader
end

function generate_dataloaders(
        sequence_length::Int,
        n_train::Int,
        n_test::Int;
        kwargs...)
    train_loader = generate_adding_data(sequence_length, n_train; kwargs...)
    test_loader = generate_adding_data(sequence_length, n_test; kwargs...)
    return train_loader, test_loader
end

struct RecurrentModel{H, C, D}
    h0::H
    rnn::C
    dense::D
end

Flux.@layer RecurrentModel trainable=(rnn, dense)

function RecurrentModel(rnn_wrapper, input_size::Int, hidden_size::Int)
    return RecurrentModel(
        zeros(Float32, hidden_size),
        rnn_wrapper(input_size => hidden_size),
        Dense(hidden_size => 1, sigmoid))
end

function (model::RecurrentModel)(inp)
    state = model.rnn(inp, model.h0)
    state = state[:, end, :]
    output = model.dense(state)
    return output
end

function train_recurrent!(epoch, train_loader, opt, model, criterion)
    total_loss = 0.0
    for (input_data, target_data) in train_loader
        input_data, target_data = CuArray(input_data), CuArray(target_data)
        grads = gradient(() -> criterion(input_data, target_data), model)
        Flux.update!(opt, model, grads[1])
        total_loss += criterion(input_data, target_data)
    end
    avg_loss = total_loss / length(train_loader)
    return avg_loss
end

function test_recurrent(epoch, test_loader, model, criterion)
    total_loss = 0.0
    for (input_data, target_data) in test_loader
        input_data, target_data = CuArray(input_data), CuArray(target_data)
        total_loss += criterion(input_data, target_data)
    end
    avg_loss = total_loss / length(test_loader)
    return avg_loss
    #println("Epoch $epoch/$num_epochs, Loss: $(round(avg_loss, digits=4))")
end

Comonicon.@main function main(rnn_wrapper;
        epochs::Int=50,
        shuffle::Bool=true,
        batchsize::Int=64,
        sequence_length::Int=1000,
        n_train::Int=500,
        n_test::Int=200,
        hidden_size::Int=20,
        learning_rate::Float64=0.01)
    train_loader, test_loader = generate_dataloaders(
        sequence_length, n_train, n_test; batchsize=batchsize, shuffle=shuffle
    )

    input_size = 2
    model = RecurrentModel(rnn_wrapper, input_size, hidden_size)
    function criterion(input_data, target_data)
        Flux.mse(
            model(input_data), reshape(target_data, 1, :)
        )
    end
    model = Flux.gpu(model)
    opt = Flux.Adam(learning_rate)

    for epoch in 1:epochs
        start_time = time()
        train_loss = train_recurrent!(epoch, train_loader, opt, model, criterion)
        test_loss = test_recurrent(epoch, test_loader, model, criterion)
        total_time = time() - start_time

        @printf "Epoch %2d: Train Loss: %.4f, Test Loss: %.4f, \
        Time: %.2fs\n" epoch train_loss test_loss total_time
    end
end
