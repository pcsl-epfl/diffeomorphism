# Diffeomorphism

arXiv preprint and reference: [Relative stability toward diffeomorphisms in deep nets indicates performance
](https://arxiv.org/abs/2105.02468)

Apply a maximum-entropy diffeomorphism to an image `img ~ [ch, n, n]`, in `PyTorch`.

Usage:

       img_diffeo = deform(img, T, cut)
       
where `T` is the temperature at which the corresponding displacement fields are drawn and `cut` is the high-frequency cutoff. 

<img src="https://github.com/leonardopetrini/diffeo-sota/blob/web/docs/diffeo_grid.png" alt="diffeo_grid_example" width="550"/>

An animated example can be found [here](https://leonardopetrini.github.io/diffeo-sota/).

The range of temperatures corresponding to _natural_ diffeomorphisms - for given `n`, `cut` - can be computed by

      Tmin, Tmax = temperature_range(n, cut)
      
To have an idea of how much distortion one gets for given `(T, cut)`, the typical displacement at the center of the image can be computed by

      delta = typical_displacement(T, cut)
      
### Diffeo Phase Space
Samples of max-entropy diffeomorphisms in the `(T, c)` phase space for an ImageNet sample. 

The region `[Tmin, Tmax]` is colored in green.

<img src="https://github.com/leonardopetrini/diffeo-sota/blob/web/docs/diffeo_phase_space.png" alt="diffeo_grid_example" width="550"/>
