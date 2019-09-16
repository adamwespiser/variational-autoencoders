module Utils

include("Model.jl")
import .Model:
  model_sample

using Images:
  Gray,
  save

function img(x)
  Gray.(rotate(reshape(x, 28, 28)))
end

function gen_images(outfile, f)
  sample = hcat(img.([model_sample(f) for i = 1:10])...)
  save(outfile, sample)
end

function gen_images(outfile, g, f, X)
  sample = hcat(img.([model_sample(g, f, X[:,:,:,i]).data for i = 1:size(X,4)])...)
  save(outfile, sample)
end

function rotate(ximg)
  xcopy = zeros(size(ximg)...)
  for i in range(1, stop = size(ximg, 1))
    for j in  range(1, stop = size(ximg, 2))
      xcopy[i,j] = ximg[j,i]
    end
  end
  xcopy
end

end
