module Model
using Random
using Distributions:Normal, Bernoulli

using Flux
using Flux.Tracker:TrackedArray, TrackedReal, track, params
export encoder, decoder

function encoder(latent_size::Int)
  Chain(
    Conv((3, 3), 1 => 32, relu, stride = (2, 2)),
    Conv((3, 3), 32 => 64, relu, stride = (2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(6 * 6 * 64, latent_size * 2)
  )
end



function decoder(latent_size::Int)
  Chain(
    Dense(latent_size, 7 * 7 * 32, relu),
    x -> reshape(x, 7, 7, 32, size(x, 2)),
    ConvTranspose((3, 3), 32 => 64, relu, stride = (2, 2), pad = (0,0)),
    ConvTranspose((3, 3), 64 => 32, relu, stride = (2,2), pad = (0,0)),
    ConvTranspose((3, 3), 32 => 1, stride = (1, 1), pad = (2,2)),
    # Somehow and extra conv filter is getting tacked on...
    # should be (28,28,1,M) but its (29,29,1,M)
    x -> x[1:28,1:28,1,1:size(x,4)]
  )
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
    x
  )
  decoder(latent_size)(x)
end


# TODO figure out what the type of X should be
# and if we want to drop the TrackedArray
# https://github.com/FluxML/Flux.jl/issues/205
# Using Flux.Tracker: data; data(X)
# However, we'll drop the grads
function split_encoder_result(X, n_latent::Int64)
  μ = X[1:n_latent, :]
  logσ = X[(n_latent + 1):(n_latent * 2), :]
  return μ, logσ
end

function split_encoder_result(X)
  n_latent = convert(Int, floort(size(X,1) / 2))
  μ = X[1:n_latent, :]
  logσ = X[(n_latent + 1):(n_latent * 2), :]
  return μ, logσ
end

## reparameterize the results of 'encoder'
# onto a Normal(0,1) distribution
# Helper Fn
function reparameterize(μ, logσ)
  eps = rand(Normal(0,1), size(μ))
  return eps * exp(logσ * 0.5) + μ
end

# KL-divergence divergence, between approximate posterior/prior
# Helper function
function kl_q_p(μ, logσ)
  return 0.5 * sum(exp.(2 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
end

# logp(x|z), conditional probability of data given latents.
# requires: f
function logp_x_z(x, z, f)
  return sum(logpdf.(Bernoulli.(f(z)), x))
end

# Monte Carlo estimator of mean ELBO using M samples.
# requires: g, f
function L̄(X, g, f, M)
  (μ̂, logσ̂) = g(X);
  return (logp_x_z(X, reparameterize.(μ̂, logσ̂), f) - kl_q_p(μ̂, logσ̂)) * 1 // M
end

# Sample from the learned model.
##modelsample() = rand.(Bernoulli.(f(z.(zeros(Dz), zeros(Dz)))))

# build_loss
#   f - decoder
#   g - split_encoder_result . encoder
#   M - n_samples
# returns loss function for X with input g/f M
function build_loss(g, f, M)
  return X -> (-L̄(X, g, f, M) + 0.01f0 * sum(x->sum(x.^2), params(f)))
end

## TODO
# specify a return s.t. for a given data shape
# we can return


end
