module Utils

include("Model.jl")
import .Model:
  model_sample

using Images:
  Gray,
  save

img(x) = Gray.(reshape(x, 28, 28))

function gen_images(outfile, f)
  sample = hcat(img.([model_sample(f) for i = 1:10])...)
  save(outfile, sample)
end

function gen_images(outfile, g, f, X)
  sample = hcat(img.([model_sample(g, f, X[:,:,:,1]) for i = 1:size(X,4)])...)
  save(outfile, sample)
end

end
