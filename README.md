# Diffeomorphism

Apply a maximum-entropy diffeomorphism to an image `img ~ [ch, n, n]`, in `PyTorch`.

Usage:

       img_diffeo = deform(img, T, cut)
       
where `T` is the temperature at which the corresponding displacement fields are drawn and `cut` is the high-frequency cutoff. 

An animated example can be found [here](https://leonardopetrini.github.io/diffeo-sota/).

The range of temperatures corresponding to _natural_ diffeomorphisms - for given `n`, `cut` - can be computed by

      Tmin, Tmax = temperature_range(n, cut)
      
To have an idea of how much distortion one gets for given `(T, cut)`, the typical displacement at the center of the image can be computed by

      delta = typical_displacement(T, cut)
      
