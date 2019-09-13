# README

#### About
A library using Julia's Flux Library to implement Variational Autoencoders (VAE).
- main.jl - run model with MINST dataset, this will be dropped later
- Model.jl the basic Model, for now it's just a basic VAE
- Dataset.jl the interface


#### Open Questions
- Can KL-Divergence and reconstruction error be better balanced?
- Can VAEs be used as a pure clustering method? In what situations does this make sense?
- Is it possible (in Julia) to reconconstruct the reverse transformation (decoder), for a given encoder?
- What's the influence of variable interaction on the quality of latent dimension embedding? 
- Can VAEs be used for columnar data to reconstruct missing input?


#### References
- [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
- [Tensorflow VAE IPython NB](python/examples/generative_examples/cvae.ipynb)
- [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
- [Flux.jl model zoo: linear VAE](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/vae.jl)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)
- [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
