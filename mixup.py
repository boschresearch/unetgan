import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

def random_boundingbox(size, lam):
    width , height = size, size

    r = np.sqrt(1. - lam)
    w = np.int(width * r)
    h = np.int(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = np.clip(x - w // 2, 0, width)
    y1 = np.clip(y - h // 2, 0, height)
    x2 = np.clip(x + w // 2, 0, width)
    y2 = np.clip(y + h // 2, 0, height)

    return x1, y1, x2, y2

def CutMix(imsize):
    lam = np.random.beta(1,1)
    x1, y1, x2, y2 = random_boundingbox(imsize, lam)
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (imsize * imsize))
    map = torch.ones((imsize,imsize))
    map[x1:x2,y1:y2]=0
    if torch.rand(1)>0.5:
        map = 1 - map
        lam = 1 - lam
    # lam is equivalent to map.mean()
    return map#, lam

###################
#  demo
###################

def cutmixdemo():
    means = 0
    for _ in range(10):
        plt.figure()
        b = CutMix(128)
        print(b.mean())
        means += b.mean()/10
        plt.imshow(b, cmap = "gray")
    print(">>>", means)
    plt.show()


#cutmixdemo()
