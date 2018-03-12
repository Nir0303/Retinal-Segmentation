import numpy as np
from PIL import Image

from scipy.special import expit


def build_image():
    test_label = np.load("cache/test_predict.npy")
    for i in range(len(test_label)):
        #print(test_label[i])

        image = test_label[i]

        """
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

        print(r[0][0], b[0][2], g[0][8])

        print(np.argmax(image,axis=2))
        print(np.max(image, axis=2))
        print(image[0][0])
        exit()
        image[...,np.argmax(image,axis=0)] =255
        """

        image = np.where(image >= 0.000000000002, 255, 0)
        # image = np.where(x, 255, 0)
        image = np.uint8(image).transpose(1, 2, 0)
        image[..., [1, 2]] = image[..., [2, 1]]


        # image[..., [1,2]] = 0

        image = Image.fromarray(image, mode='RGB')
        image.save('data/predict/label_{}.png'.format(i))

build_image()