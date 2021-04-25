import random

import numpy as np
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import matplotlib.pyplot as plt


def augmentation_func(img):
    print("elo")
    #img = rotate(img, angle=random.randint(-90,90))
    #tf = AffineTransform(shear=-0.5)
    #img = transform.warp(img, tf, order=1, preserve_range=True, mode='wrap')
    if random.randint(0,1):
        img = np.flipud(img)
    if random.randint(0, 1):
        img = np.fliplr(img)
    img = random_noise(img, var=random.uniform(0, 0.0002))
    img = img + (random.randint(-20,20) / 255)
    img = img * random.uniform(0.9,1.3)

    return img