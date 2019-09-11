
include("Dataset.jl")
import .Dataset: get_MINST, TrainTest

include("Model.jl")
import .Model:
  encoder,
  decoder,
  split_encoder_result,
  random_sample_decode,
  reparameterize


dataset = get_MINST()

n_sample = 100
n_latent = 10
X = convert(Array{Float64,3},dataset.train_x[:,:,1:n_sample])
enc_model = encoder(n_latent)
X_transformed = enc_model(reshape(X, 28, 28, 1, n_sample))

x_mean, x_std = split_encoder_result(X_transformed, n_latent)

dec_model = decoder(n_latent)
Xp = dec_model(x_mean)
