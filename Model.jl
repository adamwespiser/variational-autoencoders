module Model
using Random, Distributions

using Flux
using Flux.Tracker:TrackedArray, TrackedReal, track
export encoder, decoder

function encoder(latent_size::Int)
  Chain(
    Conv((3, 3), 1 => 32, relu, stride = (2, 2)),
    Conv((3, 3), 32 => 64, relu, stride =(2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(6 * 6 * 64, latent_size * 2)
  )
end

function decoder(latent_size::Int)
  Chain(
    Dense(latent_size, 7 * 7 * 32, relu),
    x -> reshape(x, 7, 7, 32, size(x, 2)),
    ConvTranspose((3, 3), 32 => 64, relu, stride = (2, 2)),
    ConvTranspose((3, 3), 64 => 32, relu, stride = (2, 2)),
    ConvTranspose((3, 3), 32 => 1, stride = (1, 1)))
end

function random_sample_decode(
    latent_size::Int64,
    samples::Int64
  )
  decoder(latent_size)(rand(Normal(1,1), (latent_size, samples)))
end

function random_sample_decode(
    latent_size::Int64,
    samples::Int64,
    X_samples
  )
  decoder(latent_size)(X_samples)
end


function reparameterize(x_mean, x_std)
  eps = rand(Normal(1,1), size(x_mean))
  return eps * exp(x_std * 0.5) + x_mean
end

# TODO figure out what the type of X should be
# and if we want to drop the TrackedArray
# https://github.com/FluxML/Flux.jl/issues/205
# Using Flux.Tracker: data; data(X)
# However, we'll drop the grads
function split_encoder_result(X, n_latent::Int64)
  means = X[1:n_latent, :]
  stdevs = X[(n_latent + 1):(n_latent * 2), :]
  return means, stdevs
end


end
