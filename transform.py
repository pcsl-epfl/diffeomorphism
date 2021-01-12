#pylint: disable=no-member, invalid-name, line-too-long
"""
To be used as an element of torchvision.transforms
"""
import torch
import diff


class Diffeo(torch.nn.Module):
    """Randomly apply a diffeomorphism to the image(s).
    The image should be a Tensor and it is expected to have [..., n, n] shape,
    where ... means an arbitrary number of leading dimensions.
    
    A random cut is drawn from a discrete Beta distribution of parameters
    alpha and beta such that
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards cutmax the distribution is)
        
    Given cut and the allowed* interval of temperatures [Tmin, Tmax], a random T is
    drawn from a Beta distribution with parameters alpha and beta such that:
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards T_max the distribution is)

    Beta ~ delta_function for s -> inf. To apply a specific value x \in [0, 1]
    in the allowed interval of T or cut, set
        - s = 1e10
        - r = x / (1 - x)

    *the allowed T interval is defined such as:
        - Tmin corresponds to a typical displacement of 1/2 pixel in the center
          of the image
        - Tmax corresponds to the highest T for which no overhangs are present.

    Args:
        sT (float):  
        rT (float): 
        scut (float):  
        rcut (float): 
        cut_min (int): 
        cut_max (int): 
        
    Returns:
        Tensor: Diffeo version of the input image(s).

    """
    

    def __init__(self, sT, rT, scut, rcut, cutmin, cutmax):
        super().__init__()
        
        self.sT = sT
        self.rT = rT
        self.scut = scut
        self.rcut = rcut
        self.cutmin = cutmin
        self.cutmax = cutmax
        
        self.betaT = torch.distributions.beta.Beta(sT - sT / (rT + 1), sT / (rT + 1), validate_args=None)
        self.betacut = torch.distributions.beta.Beta(scut - scut / (rcut + 1), scut / (rcut + 1), validate_args=None)
    
    def forward(self, img):
        """
        Args:
            img (Tensor): Image(s) to be 'diffeomorphed'.

        Returns:
            Tensor: Diffeo image(s).
        """
        
        # image size
        n = img.shape[-1]
        
        cut = (self.betacut.sample() * (self.cutmax + 1 - self.cutmin) + self.cutmin).int().item()
        T1, T2 = diff.temperature_range(n, cut)
        T = (self.betaT.sample() * (T2 - T1) + T1)
        
        return diff.deform(img, T, cut)
    

    def __repr__(self):
        return self.__class__.__name__ + f'(sT={self.sT}, rT={self.rT}, scut={self.scut}, rcut={self.rcut}, cutmin={self.cutmin}, cutmax={self.cutmax})'
