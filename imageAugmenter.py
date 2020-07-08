import cv2
import imutils
import os
from random import random
import math
import skimage.io as io
from skimage.transform import rotate, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import numpy as np
from skimage import exposure

if __name__ == '__main__':
    base_dir = os.path.join('RMDFDATA2')
    for i in ['face', 'mask']:
        count = 0
        folderPath = os.path.join(base_dir, i)
        # os.mkdir(os.path.join(base_dir,str(i)+'2'))
        for images in os.listdir(folderPath):
            count += 1
            file_name, file_extension = images.split('.')
            newFileName = file_name + '_' + str(count) + '.' + file_extension
            image = io.imread(os.path.join(folderPath, images))
            if random() > 0.5:
                # Rotation
                degree = int(math.ceil(random() * 50) + 1)
                print(degree)
                image = rotate(image, angle=degree, mode='wrap')
            if random() < 0.5:
                # Gaussian
                image = gaussian(image, sigma=1.3, multichannel=True)
            if random()>0.5:
                image = np.fliplr(image)
            if random()<0.5:
                image = random_noise(image, var=(random()-0.35) ** 2)
            if random()>0.5:
                image = exposure.adjust_gamma(image,gamma=1.3,gain=1)
            io.imsave(os.path.join(folderPath, newFileName), image)
