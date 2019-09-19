
include("Dataset.jl")
import .Dataset:
  get_MINST,
  TrainTest,
  split_X_by_batches,
  test_viz_idx,
  min_distance_digits

include("Model.jl")
import .Model:
  encoder,
  decoder,
  split_encoder_result,
  create_vae

include("Utils.jl")
import .Utils:
  gen_images,
  img,
  safe_img_save

using Test
using Flux
using Flux.Tracker: TrackedReal
using Printf
using Images

function experiment1()
  # CONFIG
  # TODO: break this out, after complete spec is determined
  n_traning_examples = 60000
  batch_size = 128
  n_latent = 50
  max_epoch_idx = 20
  output_dir = joinpath(pwd(), "output/1/")
  if !isdir(output_dir) # make output directory, (git ignored)
    mkdir(output_dir)
  end

  # DATASET
  dataset = get_MINST()
  X_batches = split_X_by_batches(dataset.train_x, batch_size)
  viz_idx = min_distance_digits()
  [@info "viz_idx: " idx for idx in viz_idx]

  x_view = dataset.test_x[:,:,:,viz_idx]
  # save test images (ignored by git)
  x_imgs_test = hcat(img.([x_view[:,:,:,i] for i = 1:size(x_view,4)])...)
  safe_img_save(joinpath(output_dir, "test.png"), x_imgs_test)

  # MODEL
  ps, loss_fn, f, g = create_vae(n_latent, batch_size)
  opt = ADAM()

  for epoch_idx = range(1, stop = max_epoch_idx)
    for (data_idx, dataset_batch) in enumerate(X_batches)
      start = time()
      Flux.train!(loss_fn, ps, [dataset_batch], opt)
      loss_test_set = loss_fn(dataset.test_x[:, :, :, 1:batch_size])
      @info @sprintf("[%d][%d] test loss : %.4f", epoch_idx, data_idx, loss_test_set)
      @info @sprintf("[%d][%d] elapsed %d(s)", epoch_idx, data_idx, time() - start)
      if data_idx % 5 == 1
        epoch_img = joinpath(output_dir, @sprintf("img_epoch=%d_batch=%d.png", epoch_idx, data_idx))
        gen_images(epoch_img, g, f, x_view)
      end
    end
  end
end

experiment1()
