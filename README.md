# [ICML 2025] Hessian Geometry of Latent Space in Generative Models

📄 This repository contains code and experiments for our paper:

**"Hessian Geometry of Latent Space in Generative Models"**  by Alexander Lobashev, Dmitry Guskov, Maria Larchenko, Mikhail Tamm 
Accepted to **ICML 2025**

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2506.10632)

![Fig0](figures/interpolation_big.png)

## 📝 Abstract

This paper presents a novel method for analyzing the latent space geometry of generative models, including statistical physics models and diffusion models, by reconstructing the Fisher information metric. 
The method approximates the posterior distribution of latent variables given generated samples and uses this to learn the log-partition function, which defines the Fisher metric for exponential families. 
Theoretical convergence guarantees are provided, and the method is validated on the Ising and TASEP models, outperforming existing baselines in reconstructing thermodynamic quantities. 
Applied to diffusion models, the method reveals a fractal structure of phase transitions in the latent space, characterized by abrupt changes in the Fisher metric. 
We demonstrate that while geodesic interpolations are approximately linear within individual phases, this linearity breaks down at phase boundaries, where the diffusion model exhibits a divergent Lipschitz constant with respect to the latent space. 
These findings provide new insights into the complex structure of diffusion model latent spaces and their connection to phenomena like phase transitions.

![Fig1](figures/fractal_lion_mount.png)
