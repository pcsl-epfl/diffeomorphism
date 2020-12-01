#pylint: disable=no-member, invalid-name, line-too-long
"""
To be used as an element of torchvision.transforms
"""
import torch
import diff


class Diffeo(torch.nn.Module):
    """Randomly apply a diffeomorphism to the image(s) with probability p (default 0.1).
    The image should be a Tensor and it is expected to have [..., n, n] shape,
    where ... means an arbitrary number of leading dimensions.
    
    TODO. Describe how to choose T and cut.
        from Mario: (0.5 / (0.28 * log(cut) + 0.7)) ** 2 < T < (0.4 * n * cut ** (-1.1)) ** 2,
        where n is the image size 
        (link to Mario's plot, T vs cut with green region and images?)

    Args:
        T (float): temperature 
        cut (int): high frequency cutoff
        p (float): probability that image(s) should be deformed
        
    Returns:
        Tensor: Diffeo version of the input image(s) with probability p and unchanged
        with probability (1-p).

    """
    

    def __init__(self, T, cut, p=.1):
        super().__init__()
        
        self.T = T
        self.cut = cut
        self.p = p

    
    def forward(self, img):
        """
        Args:
            img (Tensor): Image(s) to be 'diffeomorphed'.

        Returns:
            Tensor: Diffeo image(s) with probability p.
        """
        if torch.rand(1) < self.p:
            return diff.deform(img, self.T, self.cut)
        return img
    

    def __repr__(self):
        return self.__class__.__name__ + f'(T={self.T}, cut={self.cut}, p={self.p})'