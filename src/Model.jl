module Model
using Random
using Distributions:Normal, Bernoulli, logpdf

using Flux
export encoder, decoder


function encoder(latent_size::Int)
  m = Chain(
    Conv((3, 3), 1 => 32, relu, stride = (2, 2)),
    Conv((3, 3), 32 => 64, relu, stride = (2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(6 * 6 * 64, latent_size * 2)
  )
  return m
end


function decoder(latent_size::Int)
  Chain(
    Dense(latent_size, 7 * 7 * 32, relu),
    x -> reshape(x, 7, 7, 32, size(x, 2)),
    ConvTranspose((3, 3), 32 => 64, relu, stride = (2, 2), pad = (0,0)),
    ConvTranspose((3, 3), 64 => 32, relu, stride = (2,2), pad = (0,0)),
    ConvTranspose((3, 3), 32 => 1, stride = (1, 1), pad = (2,3,2,3)),
  )
end


# Requires the first dimension to be a Dense layer
function model_sample(f)
  latent_size = size(f[1].W)[2]
  zero_input = zeros(Float32, latent_size)
  rand.(Bernoulli.(sigmoid.(f(reparameterize.(zero_input, zero_input)))))
end

# overloaded version encodes/decodes X_samples
function model_sample(g, f, X_samples)
  xs = reshape(X_samples[1:28, 1:28, 1, 1], 28, 28, 1, 1)
  (μ̂, logσ̂) = split_encoder_result(g(xs))
  #rand.(Bernoulli.(sigmoid.(f(reparameterize.(μ̂, logσ̂)))))
  sigmoid.(f(reparameterize.(μ̂, logσ̂)))
end


function split_encoder_result(X, n_latent::Int64)
  μ = X[1:n_latent, :]
  logσ = X[(n_latent + 1):(n_latent * 2), :]
  return μ, logσ
end


function split_encoder_result(X)
  n_latent = convert(Int, floor(size(X,1) / 2))
  μ = X[1:n_latent, :]
  logσ = X[(n_latent + 1):(n_latent * 2), :]
  return μ, logσ
end


function reparameterize(μ :: T, logσ :: T) where {T}
  """
  reparameterize the results of 'encoder'
  onto a Normal(0,1) distribution,
  we could also pass "Float32" here instead, which would
  provide a uniform Distribution
  """
  return rand(Normal(0,1)) * exp(logσ * 0.5f0) + μ
end


function kl_q_p(μ :: T, logσ :: T) where {T}
  """"
  KL-divergence divergence, between approximate posterior/prior
  This is the KLD between a
  1) distribution of with means μ, variance logσ, and importanty, an
  identity covariance matrix, and
  2) A standard normal distribution Normal(μ = 0,σ = 1)
  """
  return 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))
end

function sigmoid(z)
  return 1.0f0 ./ (1.0f0 .+ exp(-z))
end

function logp_x_z(x, z, f)
  """
  calculates log  P( X = x | Z = z ), reconstruction error
  or: the conditional probability of data(digit image) given latent(z).
  Our output is an image, but we are creating a distribution
  using the sigmoid function to normalize decoder outputs
  then passing it into a Bernoulli distribution.
  logpdf(Distribution from decodor, image pixel) ...
  => logpdf(Bernoulli(0.5f0),{0,1}) -> {-0.693, -0.693}
  => summate over all of these values
 """
  return sum(logpdf.(Bernoulli.(sigmoid.(f(z))), x))
end


# Monte Carlo estimator of mean ELBO using M samples.
function L̄(X, g, f, M :: Int64)
  (μ̂, logσ̂) = split_encoder_result(g(X))
  return (logp_x_z(X, reparameterize.(μ̂, logσ̂), f) - kl_q_p(μ̂, logσ̂)) * 1 // M
end


# returns loss function for X with input g/f M
function build_loss(g, f, M)
  return X -> (-L̄(X, g, f, M) + 0.01f0 * sum(x->sum(x.^2), params(f)))
end


## Interface
function create_vae(latent_size :: Int64, M :: Int64)
  f = decoder(latent_size)
  g = encoder(latent_size)
  ps = params(f, g)
  loss = build_loss(g, f, M)
  return ps, loss, f, g
end

end
